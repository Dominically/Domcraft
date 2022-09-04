use std::{ops::{Range, Deref}, sync::{RwLock, Arc, mpsc::Sender}, mem, cmp::Ordering};

use itertools::iproduct;
use noise::{Perlin, NoiseFn};

use super::{chunk::{Chunk, ChunkMeshData}, player::PlayerPosition, chunk_worker_pool::ChunkType};

pub const CHUNK_SIZE: usize = 16;
pub const HEIGHTMAP_SIZE: usize = CHUNK_SIZE*CHUNK_SIZE;
pub const CHUNK_LENGTH: usize = HEIGHTMAP_SIZE*CHUNK_SIZE;
pub const CHUNK_RANGE: Range<usize> = 0..CHUNK_SIZE;

pub type SurfaceHeightmap = [i32; HEIGHTMAP_SIZE];

pub struct ChunkedTerrain {
  columns: Vec<ChunkColumn>, //Sorted in x ascending, then z ascending,
  chunk_id_bounds: [[i32; 3]; 2],
  player_last_chunk_id: [i32; 3], //The last Chunk ID of the player.
  render_distance: u32,
  worker_pool_sender: Sender<ChunkType>,
  gen: Perlin
}

impl ChunkedTerrain {
  pub fn new(player_position: PlayerPosition, render_distance: u32, worker_pool_sender: Sender<ChunkType>) -> Self {
    let player_chunk_id = player_position.block_int.map(|val| val/CHUNK_LENGTH as i32);
    let chunk_id_bounds: [[i32; 3]; 2] = [
      player_chunk_id.map(|chk| chk-render_distance as i32).into(),
      player_chunk_id.map(|chk| chk+render_distance as i32).into()
    ];

    let gen = Perlin::new();
    
    let columns: Vec<ChunkColumn> = iproduct!(
      chunk_id_bounds[0][0]..chunk_id_bounds[1][0], 
      chunk_id_bounds[0][2]..chunk_id_bounds[1][2])
      .map(|(cx, cz)| {
        let mut column = ChunkColumn::new(&gen, [cx, cz]);
        for cy in chunk_id_bounds[0][1]..chunk_id_bounds[1][1] { //Iterate vertically
          let chunk = Chunk::new(&gen, [cx, cy, cz], column.height_map);
          column.chunks.push(Arc::new(RwLock::new(chunk)));
        }
        column
      }).collect();
    
    
    Self {
      columns,
      render_distance,
      worker_pool_sender,
      chunk_id_bounds,
      player_last_chunk_id: player_chunk_id.into(),
      gen
    }
  }

  pub fn update_player_position(&mut self, player_position: PlayerPosition) { //Similar to the new() function but uses existing chunks if necessary.
    let player_chunk_id: [i32; 3] = player_position.block_int.map(|val| val / CHUNK_SIZE as i32).into();
    if player_chunk_id == self.player_last_chunk_id { //Skip the update if the player has not moved far enough to update the world.
      return;
    }

    let new_bounds: [[i32; 3]; 2] = [
      player_chunk_id.map(|chk| chk-self.render_distance as i32).into(),
      player_chunk_id.map(|chk| chk+self.render_distance as i32).into()
    ];

    let new_columns = Vec::with_capacity(self.render_distance.pow(2) as usize * 2); //Estimate columns by squaring the render distance
    let old_columns = mem::replace(&mut self.columns, new_columns); //Replace the old column list with the new one.

    let mut old_column_iter = iproduct!( //Create an iterator of old columns.
      self.chunk_id_bounds[0][0]..self.chunk_id_bounds[1][0],
      self.chunk_id_bounds[0][2]..self.chunk_id_bounds[1][2]
    ).zip(old_columns.into_iter());
    let mut next_old_column = old_column_iter.next(); //Store the next old column.

    for (ncx, ncz) in iproduct!(
      new_bounds[0][0]..new_bounds[1][0],
      new_bounds[0][2]..new_bounds[1][2]
    ) {
      
    }
  }

  pub fn get_meshes(&self) -> Vec<([i32; 3], ChunkMeshData)> {
    let mut meshes = Vec::new();
    for col in self.columns.iter() {
      for chunk in col.chunks.iter() {
        let chunk = chunk.read().unwrap();
        if let Some(mesh_data) = chunk.get_mesh() {
          let m = (
            chunk.get_id(),
            mesh_data
          );
          meshes.push(m);
        }
      }
    }

    meshes
  }

  pub fn get_column_at_mut(&mut self, column_id: &[i32; 2]) -> Option<&mut ChunkColumn> {
    let cib = &self.chunk_id_bounds;
    if (cib[0][0]..cib[1][0]).contains(&column_id[0]) && //Bounds check.
      (cib[0][2]..cib[1][2]).contains(&column_id[1]) 
    {
      let rel_chunk_id = [
        column_id[0] - cib[0][0],
        column_id[1] - cib[0][2]
      ];

      self.columns.get_mut(
        (rel_chunk_id[0] * (cib[1][0] - cib[0][0]) +
        rel_chunk_id[1]) as usize
      )
    } else {
      None
    }
  }

  pub fn get_chunk_at(&self, chunk_id: &[i32; 3]) -> Option<&Arc<RwLock<Chunk>>> {
    let cib = &self.chunk_id_bounds;
    if (cib[0][0]..cib[1][0]).contains(&chunk_id[0]) && //Bounds check. Again.
      (cib[0][1]..cib[1][1]).contains(&chunk_id[1]) &&
      (cib[0][2]..cib[1][2]).contains(&chunk_id[2]) 
    {
      let rel_pos = [ //Convert to relative position.
        chunk_id[0] - cib[0][0],
        chunk_id[1] - cib[0][1],
        chunk_id[2] - cib[0][2],
      ];

      Some(&self.columns[(
        rel_pos[0] * (cib[1][0] - cib[0][0]) + //relative x * x size + relative z
        rel_pos[2]
      ) as usize].chunks[rel_pos[1] as usize])
    } else {
      None
    }
  }

  pub fn gen_block_vis(&mut self) {
    //Generate block visibility.
    for col in self.columns.iter() {
      for chunk in col.chunks.iter() {
        let [x, y, z] = chunk.read().unwrap().get_id();
        let surrounding_chunk_locks = [
          [1i32, 0, 0],
          [-1, 0, 0],
          [0, 1, 0],
          [0, -1, 0],
          [0, 0, 1],
          [0, 0, -1]
        ].map(|[ox, oy, oz]| {
          self.get_chunk_at(&[x + ox, y + oy, z + oz]).map(|chk| chk.read().unwrap())
        });

        //TODO wait for .each to actually become usable.
        let surrounding_chunk_refs = [ //it works
          surrounding_chunk_locks[0].as_deref(),
          surrounding_chunk_locks[1].as_deref(),
          surrounding_chunk_locks[2].as_deref(),
          surrounding_chunk_locks[3].as_deref(),
          surrounding_chunk_locks[4].as_deref(),
          surrounding_chunk_locks[5].as_deref(),
        ];
        chunk.write().unwrap().gen_block_vis(surrounding_chunk_refs);
      }
    }

  }

  //Call chunk updates.
  pub fn send_chunk_update(&self) {
    for col in self.columns.iter() {
      for chunk in col.chunks.iter() {
        if chunk.read().unwrap().needs_updating() { //Only update the chunk if it needs it.
          self.worker_pool_sender.send(chunk.clone()).unwrap();
        }
      }
    }
  }
}

/// A column of chunks. Includes the heightmap for the chunk.
struct ChunkColumn {
  pub chunks: Vec<Arc<RwLock<Chunk>>>,
  pub height_map: SurfaceHeightmap
}

impl ChunkColumn {
  fn new(gen: &Perlin, chunk_xz: [i32; 2]) -> Self {
    let noise_coords = chunk_xz.map(|val| (val*CHUNK_SIZE as i32) as f64);
    
    let mut height_map: SurfaceHeightmap = [0i32; HEIGHTMAP_SIZE];
    for ((x,z), hm) in iproduct!(CHUNK_RANGE, CHUNK_RANGE).zip(height_map.iter_mut()) {
      *hm = (gen.get([
        (noise_coords[0] + x as f64) / 30.0,
        (noise_coords[1] + z as f64) / 30.0
      ]) * 5.0 + 10.0) as i32;
    }

    Self {
      chunks: Vec::new(),
      height_map
    }
  }
}
 