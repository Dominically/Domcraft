use std::{sync::{Mutex, Arc}, ops::Range};

use bytemuck_derive::{Zeroable, Pod};
use itertools::iproduct;
use noise::{Perlin, NoiseFn};
use wgpu::{Device, Queue};

use crate::renderer::buffer::{GenericBuffer, GenericBufferType};

use super::{block::{Block, BlockSideVisibility, BlockSide}, chunkedterrain::{SurfaceHeightmap, CHUNK_LENGTH, CHUNK_SIZE, CHUNK_RANGE}};

pub struct Chunk {
  chunk_id: [i32; 3],
  blocks: Vec<Block>,
  block_vis: Option<Vec<BlockSideVisibility>>,
  mesh: Mutex<Option<ChunkMesh>>,
  mesh_state: Mutex<MeshUpdateState>
}

struct ChunkMesh {
  vertex_buffer: GenericBuffer<ChunkVertex>,
  index_buffer: GenericBuffer<u32>,
}

pub struct ChunkMeshData {
  pub vertex_buffer: (Arc<wgpu::Buffer>, u64),
  pub index_buffer: (Arc<wgpu::Buffer>, u64),
}

enum MeshUpdateState {
  Outdated,
  Updating,
  Ready
}

impl Chunk {
  /// Generate a new chunk. 
  pub fn new(gen: &Perlin, chunk_id: [i32; 3], surface_heightmap: &SurfaceHeightmap) -> Self {
    let chunk_pos = chunk_id.map(|chk| {
      chk*CHUNK_SIZE as i32
    });

    
    let mut blocks = Vec::<Block>::with_capacity(CHUNK_LENGTH);
    for (x, y, z) in block_iterator() {
      let surface_level = surface_heightmap[x*CHUNK_SIZE + z];
      let actual_pos = [
        chunk_pos[0] + x as i32,
        chunk_pos[1] + y as i32,
        chunk_pos[2] + z as i32
      ];

      let noise_value = NoiseFn::<[f64; 3]>::get(gen, actual_pos.map(|val| val as f64 / 60.0));
      let is_cave = noise_value > 0.5;

      let block = if is_cave {
        Block::Air
      } else if actual_pos[1] == surface_level { //Surface
        Block::Grass
      } else if actual_pos[1] < surface_level {
        Block::Stone
      } else {
        Block::Air
      };

      

      blocks.push(block);
    }

    Self {
      chunk_id,
      blocks,
      block_vis: None,
      mesh: Mutex::new(None),
      mesh_state: Mutex::new(MeshUpdateState::Outdated)
    }
  }

  /// Gets the block at the chunk-relative location. 
  pub fn get_block_at(&self, x: isize, y: isize, z: isize) -> Option<Block> {
    const CHUNK_SIZE:isize = 16isize;
    const CHUNK_RANGE: Range<isize> = 0..CHUNK_SIZE;
    if CHUNK_RANGE.contains(&x) && CHUNK_RANGE.contains(&y) && CHUNK_RANGE.contains(&z) {
      Some(*self.blocks.get((x*CHUNK_SIZE*CHUNK_SIZE + y*CHUNK_SIZE + z) as usize).unwrap())
    } else {
      None
    }
  }

  /// Gets the surrounding blocks, or none if they are out of bounds.
  pub fn get_surrounding_blocks_of(&self, x: usize, y: usize, z: usize) -> [Option<Block>; 6] {
    let [x, y, z] = [x as isize, y as isize, z as isize];
    [
      self.get_block_at(x+1, y, z),
      self.get_block_at(x-1, y, z),
      self.get_block_at(x, y+1, z),
      self.get_block_at(x, y-1, z),
      self.get_block_at(x, y, z+1),
      self.get_block_at(x, y, z-1),
    ]
  }

  ///Generates the visibility for blocks. The adjacent chunks correspond to BlockSide for their direction.
  pub fn gen_block_vis(&mut self, adjacent_chunks: [Option<&Chunk>; 6]) { //TODO make multithreaded so it runs on different cores.
    let mut surface_visibility = Vec::<BlockSideVisibility>::with_capacity(self.blocks.len());
    for ((x, y, z), block_ref) in block_iterator().zip(self.blocks.iter()) {
      let block = *block_ref;
      if let Block::Air = block {
        surface_visibility.push(BlockSideVisibility::new(false));
        continue;
      }

      let surroundings = self.get_surrounding_blocks_of(x, y, z);
      let mut vis = BlockSideVisibility::new(false);

      for (index, block) in surroundings.into_iter().enumerate() { //Iterate each surrounding block.
        let side = BlockSide::try_from(index as u8).unwrap(); //Convert the index to BlockSide.
        match block {
            Some(block) => vis.set_visible(side, block.is_translucent()), //Block is within chunk.
            None => { //Adjacent block is outisde chunk.
              // Chunk relative position.
              let rel_pos = match side {
                BlockSide::Right => [1, y, z],
                BlockSide::Left => [CHUNK_SIZE - 1, y, z],
                BlockSide::Above => [x, 1, z],
                BlockSide::Below => [x, CHUNK_SIZE - 1, z],
                BlockSide::Back => [x, y, 1],
                BlockSide::Front => [x, y, CHUNK_SIZE - 1],
              };
              
              let block: Option<Block> = adjacent_chunks[index].and_then(|chunk| chunk.get_block_at(rel_pos[0] as isize, rel_pos[1] as isize, rel_pos[2] as isize));

              let translucent = match block {
                Some(block) => block.is_translucent(),
                None => false,
              };

              vis.set_visible(side, translucent);
            },
        }
      }
      surface_visibility.push(vis);
    }
    
    self.block_vis = Some(surface_visibility);
  }

  /// Update the vertex buffer. gen_block_vis must be called at least once before this is called. This should only be called if the vertex state is outdated.
  pub fn update_vertices(&self, device: &Device, queue: &Queue) { //Generate a vertex buffer for the chunk.
    // { //Separate scope for mutex lock.
    //   let mut state_lock = self.mesh_state.lock().unwrap();
    //   match *state_lock {
    //     MeshUpdateState::Outdated => *state_lock = MeshUpdateState::Updating,
    //     _ => return //Skip if it is not pending.
    //   }
    // }

    *self.mesh_state.lock().unwrap() = MeshUpdateState::Updating;

    let block_vis = self.block_vis.as_ref().expect("Please call gen_block_vis before generating vertices.");
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let chunk_pos = self.chunk_id.map(|val| val * CHUNK_SIZE as i32);
    
    const WINDING_ORDER: [u32; 6] = [0, 1, 2, 2, 3, 0];

    for ((x, y, z), (block, block_visibility)) in block_iterator().zip(self.blocks.iter().zip(block_vis)) {
      if block_visibility.is_invisible() {continue}; //Skip invisible blocks.
      let colour = block.get_colour();

      for side_i in 0..6 { //Corresponds to BlockSide values.
        let side = BlockSide::try_from(side_i).unwrap();

        if !block_visibility.get_visible(side) {continue}; //Skip this side if it is not visible.

        let normal = side.get_face_normal(); //get the face normal.
        let starting_index = vertices.len() as u32;
        
        for winding_index in WINDING_ORDER {
          let index = starting_index + winding_index;
          indices.push(index);
        }

        side.get_face_offset_vectors().iter().for_each(|vec| {

          let v_pos = [vec[0] + x as f32, vec[1] + y as f32, vec[2] + z as f32];
          vertices.push(ChunkVertex {
            absolute_position: chunk_pos.clone(),
            relative_position: v_pos,
            colour,
            normal
          });
        });
      }
    }

    //Update buffer with results.
    {
      let mut mesh_lock = self.mesh.lock().unwrap();
      match mesh_lock.as_mut() {
        Some(mesh) => {
          mesh.vertex_buffer.update(device, queue, &vertices);
          mesh.index_buffer.update(device, queue, &indices);
        },
        None => {

          *mesh_lock = Some(
            ChunkMesh {
                vertex_buffer: GenericBuffer::new(device, queue, GenericBufferType::Vertex, &vertices, 400),
                index_buffer: GenericBuffer::new(device, queue, GenericBufferType::Index, &indices, 600),
            }
          );
        },
      }
      
      *self.mesh_state.lock().unwrap() = MeshUpdateState::Ready; //Set state to ready.
    }
  }

  //Returns the vertex and index buffer.
  pub fn get_mesh(&self) -> Option<ChunkMeshData> {
    self.mesh.lock().unwrap().as_ref().map(|mesh| {
      ChunkMeshData {
        vertex_buffer: (mesh.vertex_buffer.get_buffer(), mesh.vertex_buffer.len() as u64),
        index_buffer: (mesh.index_buffer.get_buffer(), mesh.index_buffer.len() as u64),
      }
    }) 
  }

  pub fn get_id(&self) -> [i32; 3] {
    self.chunk_id.clone()
  }

  pub fn needs_updating(&self) -> bool {
    match *self.mesh_state.lock().unwrap() {
      MeshUpdateState::Outdated => true,
      _ => false,
    }
  }
}

fn block_iterator() -> impl Iterator<Item = (usize, usize, usize)> {
  iproduct!(CHUNK_RANGE, CHUNK_RANGE, CHUNK_RANGE)
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct ChunkVertex {
  absolute_position: [i32; 3],
  relative_position: [f32; 3],
  colour: [f32; 3],
  normal: [f32; 3]
}