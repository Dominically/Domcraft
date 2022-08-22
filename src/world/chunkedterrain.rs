use itertools::iproduct;
use noise::{Perlin, NoiseFn};

use super::chunk::Chunk;

pub const CHUNK_SIZE: usize = 16;
pub const HEIGHTMAP_SIZE: usize = CHUNK_SIZE*CHUNK_SIZE;
pub const CHUNK_LENGTH: usize = HEIGHTMAP_SIZE*CHUNK_SIZE;

pub type SurfaceHeightmap = [i32; HEIGHTMAP_SIZE];

pub struct ChunkedTerrain {
  columns: Vec<ChunkColumn>,
  render_distance: u32
}

impl ChunkedTerrain {
  fn new(player_position: [f64; 3]/*Change player position to fixed precision */, render_distance: u32) -> Self {
    let chunk_list = generate_chunk_list(player_position, render_distance);

    let columns: Vec<ChunkColumn> = Vec::new();
    
    todo!()
  }
}

/// A column of chunks. Includes the heightmap for the chunk.
struct ChunkColumn {
  pub chunk_id_xz: [i32; 2], //The x and z of the chunk ids.
  pub chunks: Vec<Chunk>,
  pub height_map: SurfaceHeightmap
}

impl ChunkColumn {
  fn new(gen: Perlin, chunk_xz: [i32; 2]) -> Self {
    let noise_coords = chunk_xz.map(|val| (val*CHUNK_SIZE as i32) as f64 / 10.0);
    let mut height_map: SurfaceHeightmap = [0i32; HEIGHTMAP_SIZE];
    for ((x,z), hm) in iproduct!(0..CHUNK_SIZE, 0..CHUNK_SIZE).zip(height_map.iter_mut()) {
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
}

fn generate_chunk_list(position: [f64; 3], render_distance: u32) -> Vec<[i32; 3]> { //Todo make circular.
  let render_distance = render_distance as i32;
  let mut chunks = Vec::new();
  let position_chunk = position.map(|val| val as i32/CHUNK_SIZE as i32);
  let boundaries = position_chunk.map(|pos| (pos - render_distance, pos + render_distance)); //TODO prevent rendering of chunks that are outside of world boundaries.

  for (cx, cy, cz) in iproduct!(boundaries[0].0 .. boundaries[0].1, boundaries[1].0..boundaries[1].1, boundaries[2].0..boundaries[2].1) {
    chunks.push([cx, cy, cz]);
  }

  chunks
}