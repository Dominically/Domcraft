use std::{ops::{Range, Deref}, sync::{RwLock, Arc, mpsc::Sender}};

use itertools::iproduct;
use noise::{Perlin, NoiseFn};

use super::{chunk::{Chunk, ChunkMeshData}, player::PlayerPosition, chunk_worker_pool::ChunkType};

pub const CHUNK_SIZE: usize = 16;
pub const HEIGHTMAP_SIZE: usize = CHUNK_SIZE*CHUNK_SIZE;
pub const CHUNK_LENGTH: usize = HEIGHTMAP_SIZE*CHUNK_SIZE;
pub const CHUNK_RANGE: Range<usize> = 0..CHUNK_SIZE;

pub type SurfaceHeightmap = [i32; HEIGHTMAP_SIZE];

pub struct ChunkedTerrain {
  columns: Vec<ChunkColumn>, //Sorted in x ascending, then z ascending
  render_distance: u32,
  worker_pool_sender: Sender<ChunkType>
}

impl ChunkedTerrain {
  pub fn new(player_position: PlayerPosition, render_distance: u32, worker_pool_sender: Sender<ChunkType>) -> Self {
    let column_list = generate_chunks_and_columns_list(player_position, render_distance);
    let gen = Perlin::new();
    
    let mut columns: Vec<ChunkColumn> = column_list.into_iter().map(|([cx, cz], chunk_ys)| {
      let mut col = ChunkColumn::new(&gen, [cx, cz]);
      let chunks: Vec<Arc<RwLock<Chunk>>> = chunk_ys.into_iter().map(|cy| {
        Arc::new(RwLock::new(Chunk::new(&gen, [cx, cy, cz], col.height_map)))
      }).collect();

      col.chunks = chunks;
      col
    }).collect();
    
    Self {
      columns,
      render_distance,
      worker_pool_sender
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
          (self.get_column_at(x + ox, z + oz), oy)
        }).map(|(opt_clmn, oy)|{
          match opt_clmn { //Mapping didn't work for some reason.
            Some(clmn) => clmn.get_chunk_at_y(y + oy).map(|chk| chk.read().unwrap()),
            None => None,
          }
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
        self.worker_pool_sender.send(chunk.clone()).unwrap();
      }
    }
  }

  fn get_column_index(&self, x: i32, z: i32) -> Result<usize, usize>{
    self.columns.binary_search_by(|col| {
      col.chunk_id_xz.cmp(&[x, z])
    })
  }

  fn get_column_at(&self, x: i32, z: i32) -> Option<&ChunkColumn> {
    self.get_column_index(x, z).ok().map(|index| self.columns.get(index).unwrap())
  }
}

/// A column of chunks. Includes the heightmap for the chunk.
struct ChunkColumn {
  pub chunk_id_xz: [i32; 2], //The x and z of the chunk ids.
  pub chunks: Vec<Arc<RwLock<Chunk>>>,
  pub height_map: SurfaceHeightmap
}

impl ChunkColumn {
  fn new(gen: &Perlin, chunk_xz: [i32; 2]) -> Self {
    let noise_coords = chunk_xz.map(|val| (val*CHUNK_SIZE as i32) as f64 / 10.0);
    let mut height_map: SurfaceHeightmap = [0i32; HEIGHTMAP_SIZE];
    for ((x,z), hm) in iproduct!(CHUNK_RANGE, CHUNK_RANGE).zip(height_map.iter_mut()) {
      *hm = (gen.get([
        noise_coords[0] + x as f64,
        noise_coords[1] + z as f64
      ]) * 5.0 + 10.0) as i32;
    }

    Self {
      chunk_id_xz: chunk_xz,
      chunks: Vec::new(),
      height_map
    }
  }

  fn get_chunk_index(&self, y: i32) -> Result<usize, usize> {
    
    self.chunks.binary_search_by(|chunk| {
      let cy = chunk.read().unwrap().get_id()[1];
      cy.cmp(&y)
    })
  }

  pub fn get_chunk_at_y(&self, y: i32) -> Option<&Arc<RwLock<Chunk>>> {
    self.get_chunk_index(y).ok().map(|index| self.chunks.get(index).unwrap())
  }
}

//Generates a list of chunk columns and chunks. Sorted by increasing z, then x (e.g. [0,0], [0,1], [0,2], [1,0], [1,1], etc...).
fn generate_chunks_and_columns_list(position: PlayerPosition, render_distance: u32) -> Vec<([i32; 2], Vec<i32>)> { //Todo make circular.
  let render_distance = render_distance as i32;
  let mut columns = Vec::new();
  let position_chunk = position.block_int.map(|pos| pos/CHUNK_SIZE as i32);
  let boundaries = position_chunk.map(|pos| (pos - render_distance, pos + render_distance)); //TODO prevent rendering of chunks that are outside of world boundaries.

  for (cx, cz) in iproduct!(boundaries[0].0..boundaries[0].1, boundaries[2].0..boundaries[2].1) {
    let mut chunks_list = Vec::new();
    for cy in boundaries[1].0..boundaries[1].1 {
      chunks_list.push(cy);
    }
    columns.push(([cx, cz], chunks_list));
  }

  columns
  
}

 