use itertools::iproduct;
use noise::Perlin;

use super::{block::{Block, BlockSideVisibility}, terrain::WorldVertex, chunkedterrain::{SurfaceHeightmap, CHUNK_LENGTH, CHUNK_SIZE}};

pub struct Chunk {
  chunk_id: [i32; 3],
  blocks: Vec<Block>,
  block_vis: Option<Vec<BlockSideVisibility>>,
  vertices: Vec<WorldVertex>,
  up_to_date: bool //Whether the block visibility is up to date.
}

impl Chunk {
  /// Generate a new chunk. 
  pub fn new(gen: Perlin, chunk_id: [i32; 3], surface_heightmap: SurfaceHeightmap) -> Self {
    let chunk_pos = chunk_id.map(|chk| {
      chk*CHUNK_SIZE as i32
    });

    
    let mut blocks = Vec::<Block>::with_capacity(CHUNK_LENGTH);
    for (x, y, z) in iproduct!(0..CHUNK_SIZE, 0..CHUNK_SIZE, 0..CHUNK_SIZE) {
      let surface_level = surface_heightmap[x*CHUNK_SIZE + z];
      let actual_pos = [
        chunk_pos[0] + x as i32,
        chunk_pos[1] + y as i32,
        chunk_pos[2] + z as i32
      ];

      let block = if actual_pos[2] == surface_level { //Surface
        Block::Grass
      } else if actual_pos[2] < surface_level {
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
      vertices: Vec::new(),
      up_to_date: false
    }
  }
}