use bytemuck_derive::{Pod, Zeroable};
use itertools::iproduct;

use super::block::{Block, BlockSideVisibility, BlockSide};

pub struct Terrain {
  blocks: Vec<Block>,
  length: usize,
  width: usize,
  height: usize,
  surface_visibility: Vec<BlockSideVisibility>,
  vertices: Vec<WorldVertex>,
  pending_vertex_update: bool
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

        for z in 0..width {
          //TESTING ONLY
          if x == 128 && y == 6 && z == 128 {
            blocks.push(Block::Air);
          } else {
            blocks.push(block);
          }
        }
      }
    }

    let mut surface_visibility = Vec::with_capacity(0);
    let mut this = Self {
      blocks,
      length,
      height,
      width,
      surface_visibility,
      pending_vertex_update: false,
      vertices: Vec::new()
    };
    
    this.update_surface_visibility();
    this.update_vertices();

    this
  }

  /// Gets a block at a certain location. Returns None if the block is out of range. WIll panic if something weird happens (this should not be possible).
  fn get_block_at(&self, x: usize, y: usize, z: usize) -> Option<Block> {
    let index = pos_to_index(x, y, z, self.length, self.height, self.width)?;
    Some(*self.blocks.get(index).expect("Index out of bounds when retrieving block."))
  }

  pub fn get_size(&self) -> (usize, usize, usize) {
    (self.length, self.height, self.width)
  }

  fn update_surface_visibility(&mut self) {
    let mut surface_visibility = Vec::with_capacity(self.blocks.len());
    for ((x, y, z), block) in self.block_iter().zip(self.blocks.iter()) {

      if let Block::Air = block { //If the block is air then set no sides to be rendered.
        surface_visibility.push(BlockSideVisibility::new(false));
        continue;
      }

      if block.is_translucent() { //Make everything transparent visible all the time.
        surface_visibility.push(BlockSideVisibility::new(true)); //Always make the block visible.
        continue;
      }

      let surrounding_blocks = [ //Offsets for surrounding blocks
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
      ].map(|offset: [usize; 3]| {
        [
          self.get_block_at(x + offset[0], y + offset[1], z + offset[2]),
          if offset[0] > x || offset[1] > y || offset[2] > z { //Prevent subtract with overflow crash.
            None
          } else {
            self.get_block_at(x - offset[0], y - offset[1], z - offset[2])
          },
        ]
      });

      let mut vis = BlockSideVisibility::new(false);

      for (block_side, adjacent_block) in surrounding_blocks.iter().flatten().enumerate() {
        let block_side = BlockSide::try_from(block_side as u8).expect("Blockside not found.");
        let adjacent_translucent: bool;
        if let Some(adjacent_block) = adjacent_block {
          adjacent_translucent = adjacent_block.is_translucent();
        } else {
          adjacent_translucent = false;
        }
        vis.set_visible(block_side, adjacent_translucent);
      }

      surface_visibility.push(vis);
    }

    self.surface_visibility = surface_visibility;
  }

  
  fn update_vertices(&mut self) {
    println!("Started updating vertices");
    const WINDING_ORDER: [usize; 6] = [0, 1, 2, 2, 3, 0];

    let mut vertices = Vec::<WorldVertex>::with_capacity(1000000); //big vec.
    for ((x, y, z), (vis, block)) in self.block_iter().zip(self.surface_visibility.iter().zip(self.blocks.iter())) {
      let uv_initial = match block {
        Block::Stone => [0.25, 0.0],
        Block::Grass => [0.0, 0.0],
        Block::Bedrock => [0.5, 0.0],
        _ => [0.75, 0.0], //Missing texture
      };
      
      for side_i in 0..6 {
        let side = BlockSide::try_from(side_i).unwrap();
        if vis.get_visible(side) {
          let vecs = side.get_face_offset_vectors().map(|vec| {
            [vec[0] + x as f32, vec[1] + y as f32, vec[2] + z as f32]
          });
          // println!("XYZ {}, {}, {}", x, y, z);
          // println!("Vecs: {:?}", vecs);
          // panic!();
          
          for vertex_index in WINDING_ORDER {
            let uv_offset = match vertex_index {
              1 => [0.25, 0.0],
              2 => [0.25, 0.25],
              3 => [0.0, 0.25],
              _ => [0.0, 0.0],
            };

            vertices.push(
              WorldVertex {
                position: vecs[vertex_index],
                uv: [uv_initial[0] + uv_offset[0], uv_initial[1] + uv_offset[1]]
              }
            );
          }
        }
      }
    }

    self.vertices = vertices;
    self.pending_vertex_update = true;

    println!("Done updating vertices of length: {:?}", self.vertices.len());
  }

  fn block_iter(&self) -> impl Iterator<Item = (usize, usize, usize)> {
    iproduct!(0..self.length, 0..self.height, 0..self.width)
  }

  pub fn get_vertex_update(&mut self) -> Option<&Vec<WorldVertex>> {
    if self.pending_vertex_update {
      self.pending_vertex_update = false;
      Some(&self.vertices)
    } else {
      None
    }
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