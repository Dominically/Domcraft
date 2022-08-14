#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Block {
  Stone,
  Grass,
  Bedrock,
  Air
}

impl Block {
  fn is_transparent(&self) -> bool {
    match self {
      Air => true,
      _ => false
    }
  }
}

#[repr(u8)]
pub enum BlockSide {
  Front = 0,
  Back = 1,
  Left = 2,
  Right = 3,
  Bottom = 4,
  Top = 5
}

/// This is so we can cull faces that we can't see.
pub struct BlockSideVisibility {
  flags: u8 //Uses bit setting because i dont want to run out of ram again.
}

impl BlockSideVisibility {
  pub fn new(default_transparent: bool) -> Self {
    let flags = if default_transparent {0x3Fu8} else {0x00u8};
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
}

#[cfg(test)]
mod tests {
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
}