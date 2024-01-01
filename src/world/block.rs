#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Block {
  Stone,
  Grass,
  Bedrock,
  PinkStuff,
  YellowStuff,
  Cloud,
  Air
}

impl Block {
  pub fn is_translucent(&self) -> bool {
    match self {
      Block::Air => true,
      _ => false
    }
  }

  pub fn get_colour(&self) -> [f32; 4] {
    match self {
      Block::Stone => [0.5, 0.5, 0.5, 1.0],
      Block::Grass => [0.3, 0.7, 0.3, 1.0],
      Block::Bedrock => [0.1, 0.1, 0.1, 1.0],
      Block::YellowStuff => [0.5, 0.5, 0.2, 1.0],
      Block::Cloud => [0.8, 0.8, 0.8, 0.5],
      _ => [1.0, 0.0, 1.0, 1.0], //MISSING COLOUR
    }
  }
}

#[repr(u8)]
#[derive(Clone, Copy)]
pub enum BlockSide {
  Right = 0,
  Left = 1,
  Above = 2,
  Below = 3,
  Back = 4,
  Front = 5,
}

impl BlockSide {
  pub fn get_face_offset_vectors(&self) -> [[f32; 3]; 4] { //Verts are wound counter-clockwise
    match self { //See cubedirections.png
        BlockSide::Right => [4, 5, 7, 6],
        BlockSide::Left => [0, 2, 3, 1],
        BlockSide::Above => [3, 2, 6, 7],
        BlockSide::Below => [0, 1, 5, 4],
        BlockSide::Back => [1, 3, 7, 5],
        BlockSide::Front => [0, 4, 6, 2],
    }.map(number_to_offset_vector)
  }

  pub fn get_face_normal(&self) -> [f32; 3] {
    match self {
        BlockSide::Right => [1.0, 0.0, 0.0],
        BlockSide::Left => [-1.0, 0.0, 0.0],
        BlockSide::Above => [0.0, 1.0, 0.0],
        BlockSide::Below => [0.0, -1.0, 0.0],
        BlockSide::Back => [0.0, 0.0, 1.0],
        BlockSide::Front => [0.0, 0.0, -1.0],
    }
  }
}

impl TryFrom<u8> for BlockSide {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
          0 => Ok(BlockSide::Right),
          1 => Ok(BlockSide::Left),
          2 => Ok(BlockSide::Above),
          3 => Ok(BlockSide::Below),
          4 => Ok(BlockSide::Back),
          5 => Ok(BlockSide::Front),
          _ => Err(())
        }
    }
}

/// This is so I can cull faces that can't be seen.
#[derive(Debug, Copy, Clone)]
pub struct BlockSideVisibility {
  flags: u8 //Uses bit setting because i dont want to run out of ram again.
}

impl BlockSideVisibility {
  pub fn new(default_visibility: bool) -> Self {
    let flags = if default_visibility {0x3Fu8} else {0x00u8};
    Self {
      flags
    }
  }

  /// Set the state of a side of a block.
  pub fn set_visible(&mut self, side: BlockSide, value: bool) {
    if value { //magic and stuff.
      self.flags |= 1u8 << side as u8;
    } else {
      self.flags &= !(1u8<< side as u8);
    }
  }

  pub fn get_visible(&self, side: BlockSide) -> bool { //more magic
    (self.flags & 1u8 << side as u8) > 0
  }

  pub fn is_invisible(&self) -> bool {
    self.flags == 0
  }
}

fn number_to_offset_vector(num: usize) -> [f32; 3] { //see cube directions.png
  [
    num&0b100 > 0,
    num&0b10 > 0,
    num&0b1 > 0,
  ].map(|val| if val {1.0} else {0.0})
}

#[cfg(test)]
mod tests {
  use crate::world::block::number_to_offset_vector;

use super::{BlockSideVisibility, BlockSide};

  #[test]
  fn test_block_side() {
    let mut vis = BlockSideVisibility::new(false);
    assert!(!vis.get_visible(BlockSide::Front)); //Test default.

    vis.set_visible(BlockSide::Front, true); //Test setting.
    assert!(vis.get_visible(BlockSide::Front));

    vis.set_visible(BlockSide::Front, false); //Test resetting.
    assert!(!vis.get_visible(BlockSide::Front));

    vis.set_visible(BlockSide::Right, true); //Test another side.
    assert!(vis.get_visible(BlockSide::Right));


  }

  #[test]
  fn test_offset_vector() {
    assert_eq!(number_to_offset_vector(1), [0.0, 0.0, 1.0]);
    assert_eq!(number_to_offset_vector(0), [0.0, 0.0, 0.0]);
    assert_eq!(number_to_offset_vector(4), [1.0, 0.0, 0.0]);
    assert_eq!(number_to_offset_vector(7), [1.0, 1.0, 1.0]);
  }
}