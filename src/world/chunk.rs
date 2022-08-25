use itertools::iproduct;
use noise::Perlin;

use super::{block::{Block, BlockSideVisibility, BlockSide, self}, terrain::WorldVertex, chunkedterrain::{SurfaceHeightmap, CHUNK_LENGTH, CHUNK_SIZE, CHUNK_RANGE}};

pub struct Chunk {
  chunk_id: [i32; 3],
  blocks: Vec<Block>,
  block_vis: Option<Vec<BlockSideVisibility>>,
}

impl Chunk {
  /// Generate a new chunk. 
  pub fn new(gen: &Perlin, chunk_id: [i32; 3], surface_heightmap: SurfaceHeightmap) -> Self {
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
    }
  }

  /// Gets the block at the chunk-relative location. 
  pub fn get_block_at(&self, x: usize, y: usize, z: usize) -> Option<Block> {
    if CHUNK_RANGE.contains(&x) && CHUNK_RANGE.contains(&y) && CHUNK_RANGE.contains(&z) {
      Some(*self.blocks.get(x*CHUNK_SIZE*CHUNK_SIZE + y*CHUNK_SIZE + z).unwrap())
    } else {
      None
    }
  }

  /// Gets the surrounding blocks, or none if they are out of bounds.
  pub fn get_surrounding_blocks_of(&self, x: usize, y: usize, z: usize) -> [Option<Block>; 6] {
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
              
              let block: Option<Block> = adjacent_chunks[index].and_then(|chunk| chunk.get_block_at(rel_pos[0], rel_pos[1], rel_pos[2]));

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
  }

  /// Gets the vertices of the chunk. gen_block_vis must be called at least once before this is called.
  pub fn get_vertices(&self) -> Vec<WorldVertex> { //Generate a vertex buffer for the chunk.
    let block_vis = self.block_vis.as_ref().expect("Please call gen_block_vis before generating vertices.");
    let vertices = Vec::new();
    for ((x, y, z), (block, block_visibility)) in block_iterator().zip(self.blocks.iter().zip(block_vis)) {
      todo!();
    }
    vertices
  }
}

fn block_iterator() -> impl Iterator<Item = (usize, usize, usize)> {
  iproduct!(CHUNK_RANGE, CHUNK_RANGE, CHUNK_RANGE)
}