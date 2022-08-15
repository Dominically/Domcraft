use bytemuck_derive::{Pod, Zeroable};
use itertools::iproduct;

use super::block::{Block, BlockSideVisibility, BlockSide};

pub struct Terrain {
  blocks: Vec<Block>,
  length: usize,
  width: usize,
  height: usize,
  surface_visibility: Vec<BlockSideVisibility>
}

impl Terrain {
  /// Generate a superflat world.
  pub fn gen_superflat(length: usize, height: usize, width: usize) -> Self {
    let mut blocks = Vec::with_capacity(length * width * height);
    for x in 0..length {
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

        for _z in 0..width {
          blocks.push(block);
        }
      }
    }

    let mut surface_visibility = Vec::<BlockSideVisibility>::with_capacity(blocks.len());

    for (index, block) in blocks.iter().enumerate() {
      if let Block::Air = block { //If the block is air then set no sides to be rendered.
        surface_visibility.push(BlockSideVisibility::new(false));
        continue;
      }

      if block.is_translucent() { //Make everything transparent visible all the time.
        surface_visibility.push(BlockSideVisibility::new(true)); //Always make the block visible.
        continue;
      }

      let surrounding_blocks = [ //Offsets for surrounding blocks
        width*height,
        width,
        1
      ].map(|offset| {
        [
          blocks.get(index + offset).map(|b| *b),
          blocks.get(index - offset).map(|b| *b), //Hopefully the overflow won't crash it. 
        ]
      });

      let mut vis = BlockSideVisibility::new(false);

      for (block_side, adjacent_block) in surrounding_blocks.iter().flatten().enumerate() {
        let block_side = BlockSide::try_from(block_side as u8).expect("Blockside not found.");
        let adjacent_translucent: bool;
        if let Some(adjacent_block) = adjacent_block {
          adjacent_translucent = adjacent_block.is_translucent();
        } else {
          adjacent_translucent = true;
        }
        vis.set_visible(block_side, adjacent_translucent);
      }

      surface_visibility.push(vis);
    }
    
    Self {
      blocks,
      length,
      height,
      width,
      surface_visibility,
    }
  }

  /// Gets a block at a certain location. Returns None if the block is out of range. WIll panic if something weird happens (this should not be possible).
  fn get_block_at(&self, x: usize, y: usize, z: usize) -> Option<Block> {
    let index = pos_to_index(x, y, z, self.length, self.height, self.width)?;
    Some(*self.blocks.get(index).expect("Index out of bounds when retrieving block."))
  }

  pub fn get_size(&self) -> (usize, usize, usize) {
    (self.length, self.height, self.width)
  }

  pub fn get_vertices(&self) -> Vec<WorldVertex> {
    const WINDING_ORDER: [usize; 6] = [0, 1, 2, 0, 2, 3];

    let mut vertices = Vec::<WorldVertex>::new();
    for ((x, y, z), vis) in iproduct!(0..self.length, 0..self.height, 0..self.width).zip(self.surface_visibility.iter()) {
      for side_i in 0..6 {
        let side = BlockSide::try_from(side_i).unwrap();
        if vis.get_visible(side) {
          let vecs = side.get_face_offset_vectors().map(|vec| {
            [vec[0] + x as f32, vec[1] + y as f32, vec[2] + z as f32]
          });
          
          for vertex_index in WINDING_ORDER {
            vertices.push(
              WorldVertex {
                position: vecs[vertex_index],
                uv: [0.0, 0.0] //TODO
              }
            );
          }
        }
      }
    }

    vertices
  }
}


fn pos_to_index(x: usize, y: usize, z: usize, length: usize, height: usize, width: usize) -> Option<usize> {
  if x < length && y < height && z < width {
    Some(x*height*width + y*width + z)
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