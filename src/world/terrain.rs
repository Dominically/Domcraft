use bytemuck_derive::{Pod, Zeroable};
use itertools::iproduct;

use super::block::{Block, BlockSideVisibility};

pub struct Terrain {
  blocks: Vec<Block>,
  length: usize,
  width: usize,
  height: usize,
  surface_visibility: BlockSideVisibility
}

impl Terrain {
  /// Generate a superflat world.
  pub fn gen_superflat(length: usize, width: usize, height: usize) -> Self {
    let mut blocks = Vec::with_capacity(length * width * height);
    for y in 0..height {
      let block = match y {
        0 => { //Ground = bedrock
          Block::Bedrock
        },
        1..=5 => { //Some stone
          Block::Stone
        },
        6 => { //Grass to top it off.
          Block::Grass
        },
        _ => { //Air for the rest
          Block::Air
        }
      };

      for _ in 0..length { //Fill y-slice with blocks.
        for _ in 0..width {
          blocks.push(block);
        }
      }
    };

    let surface_visibilities = Vec::<BlockSideVisibility>::with_capacity(blocks.len());

    for (index, (y, z, x)) in block_iter(height, width, length) {
      
    }

    Self { 
      blocks,
      length,
      width,
      height,
      surface_visibility: todo!()
    }
  }

  /// Gets a block at a certain location. Returns Err() if the block is out of range.
  fn get_block_at(&self, x: usize, y: usize, z: usize) -> Option<Block> {
    Some(*self.blocks.get(pos_to_index(x, y, z, self.length, self.height, self.width)?).expect("pos_to_index gave a bad value. This should not be possible!")) //TODO change to unsafe get_unchecked.
  }

  pub fn get_size(&self) -> (usize, usize, usize) {
    (self.length, self.height, self.width)
  }

  pub fn get_vertices(&self) -> Vec<WorldVertex> {
    let mut vertices = Vec::<WorldVertex>::new();
    let iterator = block_iter(self.height, self.width, self.length);
    for (index, (y, z, x)) in iterator {
      todo!();
    }

    vertices
  }
}

/// Is in order of y z x.
fn block_iter(height: usize, width: usize, length: usize) -> impl Iterator<Item = (usize, (usize, usize, usize))> { //this probably shouldn't be in the impl block.
let product = iproduct!(0..height, 0..width, 0..length);
product.enumerate()
}


fn pos_to_index(x: usize, y: usize, z: usize, length: usize, height: usize, width: usize) -> Option<usize> {
  if x < length && y < height && z < width {
    Some(y*width*length + z*length + x)
  } else {
    None
  }
}



#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct WorldVertex {
  pub position: [f32; 3],
  pub uv: [f32; 2]
}

#[cfg(test)]
mod tests {
    use super::{Block, Terrain};

  #[test]
  fn test_superflat() { //Test world gen.
    let world = Terrain::gen_superflat(256, 256, 256);

    assert_eq!(world.get_block_at(0, 0, 0), Some(Block::Bedrock));
    assert_eq!(world.get_block_at(45, 0, 20), Some(Block::Bedrock));
    assert_eq!(world.get_block_at(300, 4, 10), None);
    assert_eq!(world.get_block_at(3, 3, 3), Some(Block::Stone));
    assert_eq!(world.get_block_at(100, 6, 255), Some(Block::Grass));
    assert_eq!(world.get_block_at(255, 255, 255), Some(Block::Air));
    assert_eq!(world.get_block_at(256, 255, 255), None)
  }
}