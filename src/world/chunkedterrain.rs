use std::{ops::Range, sync::{Arc, mpsc::Sender}, mem, cmp::Ordering};

use cgmath::{num_traits::Signed, InnerSpace, Vector3};
use fixed::traits::Fixed;
use itertools::{iproduct, Itertools};
use noise::{Perlin, NoiseFn, Seedable};
use num_iter::{range_step_inclusive, RangeStepInclusive};

use crate::{world::chunk::ChunkDataView, util::{FPVector, Fixed64}};

use super::{chunk::{Chunk, ChunkMeshData, ChunkStateStage, ADJACENT_OFFSETS}, player::HitBox, chunk_worker_pool::{ChunkTask, ChunkTaskType}, block::{Block, BlockSide, BlockSideVisibility}};

pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_SIZE_I32: i32 = CHUNK_SIZE as i32;
pub const HEIGHTMAP_SIZE: usize = CHUNK_SIZE*CHUNK_SIZE;
pub const CHUNK_LENGTH: usize = HEIGHTMAP_SIZE*CHUNK_SIZE;
pub const CHUNK_RANGE: Range<usize> = 0..CHUNK_SIZE;

pub type SurfaceHeightmap = [i32; HEIGHTMAP_SIZE];

pub struct ChunkedTerrain {
  columns: Vec<ChunkColumn>, //Sorted in x ascending, then z ascending,
  chunk_id_bounds: [[i32; 3]; 2],
  player_last_chunk_id: Vector3<i32>, //The last Chunk ID of the player.
  render_distance: u32,
  worker_pool_sender: Sender<ChunkTask>,
  gen: Arc<Perlin>,
  chunk_gc: Sender<Arc<Chunk>>
}

struct BlockVisArea {
  data: Vec<BlockSideVisibility>,
  size: Vector3<usize>
}

impl ChunkedTerrain {
  pub fn new(player_position: FPVector, render_distance: u32, worker_pool_sender: Sender<ChunkTask>, chunk_gc: Sender<Arc<Chunk>>) -> Self {
    let player_chunk_id = Self::pos_to_chunk_id(player_position.get_int());
    let chunk_id_bounds: [[i32; 3]; 2] = [
      player_chunk_id.map(|chk| chk-render_distance as i32).into(),
      player_chunk_id.map(|chk| chk+render_distance as i32).into()
    ];
    
    let gen = Arc::new(Perlin::new().set_seed(7355608));
    
    let columns: Vec<ChunkColumn> = iproduct!(
      chunk_id_bounds[0][0]..chunk_id_bounds[1][0], 
      chunk_id_bounds[0][2]..chunk_id_bounds[1][2])
      .map(|(cx, cz)| {
        let mut column = ChunkColumn::new(&gen, [cx, cz]);
        for cy in chunk_id_bounds[0][1]..chunk_id_bounds[1][1] { //Iterate vertically
          column.chunks.push(make_new_chunk([cx, cy, cz]));
        }
        column
      }).collect();
    
    
    Self {
      columns,
      render_distance,
      worker_pool_sender,
      chunk_id_bounds,
      player_last_chunk_id: player_chunk_id.into(),
      gen,
      chunk_gc
    }
  }

  //Returns true if the chunk vertices need to be regenerated.
  pub fn update_player_position(&mut self, player_position: &FPVector) -> bool { //Similar to the new() function but uses existing chunks if necessary.
    let player_chunk_id = Self::pos_to_chunk_id(player_position.get_int());
    if player_chunk_id == self.player_last_chunk_id { //Skip the update if the player has not moved far enough to update the world.
      return false;
    }

    let new_bounds: [[i32; 3]; 2] = [
      player_chunk_id.map(|chk| chk-self.render_distance as i32).into(),
      player_chunk_id.map(|chk| chk+self.render_distance as i32).into()
    ];

    let needs_regen = [ //Compare new bounds to old bounds to see if edges of chunks need regenerating.
      (new_bounds[0][0] < self.chunk_id_bounds[0][0], new_bounds[1][0] > self.chunk_id_bounds[1][0]), //X expansion,
      (new_bounds[0][1] < self.chunk_id_bounds[0][1], new_bounds[1][1] > self.chunk_id_bounds[1][1]), //Y expansion
      (new_bounds[0][2] < self.chunk_id_bounds[0][2], new_bounds[1][2] > self.chunk_id_bounds[1][2]), //Z expansion
    ];


    let regen = [0, 1, 2].map(|i| {
      let (a, b) = (needs_regen[i].0, needs_regen[i].1);
      if a && b {
        ChunkRegenCoord::Both(self.chunk_id_bounds[0][i], self.chunk_id_bounds[1][i] - 1)
      } else if a { //Value is decreasing
        ChunkRegenCoord::One(self.chunk_id_bounds[0][i])
      } else if b { //Value is increasing
        ChunkRegenCoord::One(self.chunk_id_bounds[1][i] - 1) //Chunk high bounds are exclusive so subtract 1.
      } else {
        ChunkRegenCoord::None
      }
    });
    

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
      while matches!(&next_old_column, Some(col) if matches!([col.0.0, col.0.1].cmp(&[ncx, ncz]), Ordering::Less)) { //Skip columns that have already passed.
        for chunk in next_old_column.unwrap().1.chunks { //Send chunks to gc so deleting them doesn't block this thread.
          self.chunk_gc.send(chunk).unwrap();
        }
        next_old_column = old_column_iter.next();
      }

      let column = match &next_old_column {
        //Column already exists.
        Some(col) if [col.0.0, col.0.1] == [ncx, ncz] => {
          let mut column = next_old_column.unwrap().1;
          next_old_column = old_column_iter.next();
          self.reuse_column(&mut column, new_bounds, [ncx, ncz], regen);
          column
        },
        //Create new column.
        _ => {
          let mut column = ChunkColumn::new(&self.gen, [ncx, ncz]);
          for ncy in new_bounds[0][1]..new_bounds[1][1] {
            let chunk = make_new_chunk([ncx, ncy, ncz]);
            column.chunks.push(chunk);
          }
          column
        },
      };
      self.columns.push(column);
    }

    self.chunk_id_bounds = new_bounds; //Update bounds.
    self.player_last_chunk_id = player_chunk_id;

    true
  }

  pub fn get_meshes(&self) -> Vec<([i32; 3], ChunkMeshData)> {
    let mut meshes = Vec::new();
    for col in self.columns.iter() {
      for chunk in col.chunks.iter() {
        if let Some(mesh_data) = chunk.get_mesh_fast() {
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

  pub fn get_chunk_at(&self, chunk_id: &[i32; 3]) -> Option<&Arc<Chunk>> {
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

  
  pub fn get_block_at(&self, pos: Vector3<i32>) -> Option<Block> {
    
    let div = pos / (CHUNK_SIZE_I32);
    let neg = pos.map(|v| if v < 0 {-1} else {0});
    let chunk_id = div+neg;


    let chunk = self.get_chunk_at(&chunk_id.into())?; //For some reason I've not used vector3s in my terrain data.

    let inner_pos = pos - (chunk_id * (CHUNK_SIZE_I32));
    let block = chunk.get_block_at(inner_pos.x, inner_pos.y, inner_pos.z)?;

    Some(block)
  }

  //Call chunk updates.
  pub fn tick_progress(&self) {
    for col in self.columns.iter() {
      for chunk in col.chunks.iter() {
        let stage = chunk.get_pending_stage();
        
        match stage {
          Some(ChunkStateStage::ChunkGen) => {
            self.send_task(ChunkTask {
              chunk: chunk.clone(),
              typ: ChunkTaskType::GenTerrain(self.gen.clone(), col.height_map.clone()),
            });
          },
          Some(ChunkStateStage::ChunkVisGen) => {
            let [idx, idy, idz] = chunk.get_id();
            let adjacent_chunks = ADJACENT_OFFSETS.map(|[ox, oy, oz]| {
              self.get_chunk_at(&[idx + ox, idy + oy, idz + oz]).map(|chunk| chunk.clone())
            });
            if !adjacent_chunks.iter().any(|chunk| { //Check if the chunk is adjacent to chunks that are still generating. If so then skip it.
              chunk.as_ref().map_or(false, |chunk| {
                chunk.get_stage() == ChunkStateStage::ChunkGen
              })
            }) { //Then send it to be processed.
              self.send_task(ChunkTask {
                chunk: chunk.clone(),
                typ: ChunkTaskType::GenBlockVis(adjacent_chunks),
              });
            }
          },
          Some(ChunkStateStage::MeshGen) => {
            self.send_task(ChunkTask {
              chunk: chunk.clone(),
              typ: ChunkTaskType::GenVertices,
            });
          },
          _ => {
            //Do nothing for now.
          },
        };
      }
    }
  }

  fn send_task(&self, task: ChunkTask) {
    if task.chunk.assign_if_waiting() {
      self.worker_pool_sender.send(task).unwrap();
    }
  }


  /**
    Tests the hitbox from its faces. Blocks that are completely inside the hitbox will not be checked.
    
    NOTE: This is (currently) not meant to be called at high frequency (many times per tick)
    and will require optimisation if this is necessary.
    This will also prevent the player from entering chunks that are not yet generated.

   */
  pub fn update_collision_info(&self, current_pos: &mut FPVector, velocity: &mut Vector3<f32>, secs: f32, hitbox: &HitBox) {
    let delta = *velocity * secs;
    let target_pos = *current_pos + FPVector::from(delta); //The position if there is no collision.


    //Calculate the absolute positions of the hitbox corners.
    let a1 = *current_pos + hitbox.lo;
    let b1 = *current_pos + hitbox.hi;
    let a2 = target_pos + hitbox.lo;
    let b2 = target_pos + hitbox.hi;

    //Work out bounds of hitbox check. This is so we can work in relative coordinates to preserve floating-point accuracy.
    let (min, max) = [a1, b1, a2, b2].into_iter().fold((a1.get_int(), b1.get_int()), |(min, max), pos| ( //Start with default positions.
      min.zip(pos.get_int(), |p, q| p.min(q)),
      max.zip(pos.get_int(), |p, q| p.max(q))
    ));

    let vis_data = self.get_block_vis_area(min, max);

    let mut min_t = Fixed64::ONE; //Tick period at first collision.
    let mut intersect_data: [Option<Fixed64>; 3] = [None; 3];

    for dir_dim in 0..3 { //Now test sides for collision.
      if delta[dir_dim] == 0.0 {continue}; //Skip if the object is not moving in this direction.

      let is_positive_dir = delta[dir_dim].is_positive(); //true = hi, false = lo
      let block_side = BlockSide::try_from((dir_dim as u8)*2 + if is_positive_dir {1} else {0}).unwrap();

      let [(a_old, /*b_old*/), (a_new,/* b_new*/)] = [*current_pos, target_pos].map(|p|(
        p + if is_positive_dir {hitbox.hi} else {hitbox.lo},
        // p + Vector3::from([0usize,1,2].map(|i| if is_positive_dir ^ (i==dir_dim) {hitbox.lo[i]} else {hitbox.hi[i]})) //Use coordinates of opposite corner except for the dimension of the side we are testing.
      ));
      
      //The orientation of these layers depends on the direction we are testing.
      
      //Convert to f32 for ease of use.
      let rel_a_old = a_old - min.into();
      let rel_a_new = a_new - min.into();
      
      let l_old = rel_a_old.inner[dir_dim];
      let l_new = rel_a_new.inner[dir_dim];

      let layers = get_layers_between(l_old, l_new, false);
      
      // if dir_dim == 1  && delta[dir_dim] < 0.0 {
      //   let lc  = layers.clone().map(|l| l.map(|v| v + min[dir_dim]).collect_vec());
      //   println!("min: {}, l_old: {}, l_new: {}, Test layers: {lc:?}", min[dir_dim],  l_old + Fixed64::from_num(min.y), l_new + Fixed64::from_num(min.y));
      // }

      if let Some(iter) = layers{
        'layer_loop: for layer in iter {
          let layer_fixed64 = Fixed64::from_num(layer);

          //TODO stop using t and min_t for colliding dimension. (probably done).
          //Add one to the layer because we are testing the block one less when the direction is negative.
          let t = (layer_fixed64 - l_old )/(l_new - l_old); //This represents the fraction of the tick where the player passes through the layer.
          
          let rel_a_pos = {
            let mut pos = rel_a_old + (rel_a_new - rel_a_old) * t; //Get the position of the face at this point in time.
            pos.inner[dir_dim] = layer_fixed64; //Set layer to full integer to prevent precision errors.
            pos
          };
          
          let rel_b_pos = {
            let mut new_pos = rel_a_pos + if is_positive_dir {hitbox.lo - hitbox.hi} else {hitbox.hi - hitbox.lo};
            new_pos.inner[dir_dim] = layer_fixed64; //Replace new_pos with old a_pos in current dim we are checking.
            new_pos
          };

          //Create iterator over test range.
          let [lx, ly, lz]: [RangeStepInclusive<i32>; 3] = rel_a_pos.inner.zip(rel_b_pos.inner, |a, b| 
            range_step_inclusive(a.int().to_num(), b.int().to_num(), if a>b {-1i32} else {1})
          ).into();

          for (rel_x, rel_y, rel_z) in iproduct!(lx, ly, lz) {
            let mut rel_usize = Vector3::from([rel_x as usize, rel_y as usize, rel_z as usize]);
            
            if !is_positive_dir { //bugfix
              rel_usize[dir_dim] -= 1;
            }

            let vis = vis_data.get_block_at(rel_usize);
            if vis.get_visible(block_side) {
              if t <= min_t {
                min_t = t;
                let dim_pos = Fixed64::from_num(min[dir_dim]) + layer_fixed64 - if is_positive_dir {hitbox.hi.inner[dir_dim]} else {hitbox.lo.inner[dir_dim]};
                
                intersect_data[dir_dim] = Some(dim_pos);
              }
              break 'layer_loop;
            }
          }
        }
      }
    }
    
    let velocity_fpv: FPVector = (*velocity).into();

    let mut new_pos = *current_pos + velocity_fpv * (min_t * Fixed64::from_num(secs)); //TODO temp for now.

    // if current_pos.inner.y >= 1.5 && new_pos.inner.y < 1.5 {
    //   println!("Passing through floor.");
    // }

    for (i_dim, i_val) in intersect_data.iter().enumerate() {
      if let Some(val) = *i_val {
        //TODO collision resolution here.
        new_pos.inner[i_dim] = val;
        velocity[i_dim] = 0.0;
      }
    }

    if min_t < Fixed64::ONE && velocity.magnitude() > 0.0{ //If tick has not been fully processed and player is still moving.
      self.update_collision_info(&mut new_pos, velocity, secs * (1.0f32 - min_t.to_num::<f32>()), hitbox);
    }

    *current_pos = new_pos;
  }

  
  ///Get visibility of blocks in a given area and store it into a (temporary) struct. This is used for hitbox testing.
  fn get_block_vis_area(&self, min: Vector3<i32>, max: Vector3<i32>) -> BlockVisArea {
    //Input validation check (this should never panic ideally).
    min.zip(max, |mn, mx| if mn > mx {panic!("Invalid range passed to get_block_vis_area.")});


    //Get positions of chunks we need.
    let [cid_min, cid_max] = [min, max].map(|vec| Self::pos_to_chunk_id(vec));
    
    //Create a range from positions.
    let cid_len = cid_min.zip(cid_max, |mn, mx| mx - mn) + Vector3::from([1i32; 3]); //Add one to length because range is inclusive.

    let mut vis_data = Vec::<BlockSideVisibility>::with_capacity(((max.x-min.x)*(max.y-min.y)*(max.z-min.z)) as usize); 
    
    let mut chunk_vis = Vec::new(); //probably not needed to use with_capacity as it won't usually be that many chunks

    for (cx, cy, cz) in iproduct!(cid_min.x..=cid_max.x, cid_min.y..=cid_max.y, cid_min.z..=cid_max.z) {
      let data = self.get_chunk_at(&[cx, cy, cz]).map_or_else(|| ChunkDataView::new_blank(), |c| c.get_data_view());
      // let data = ChunkDataView::new_blank();
      chunk_vis.push(data);
    }
    for (x, y, z) in iproduct!(min.x..=max.x, min.y..=max.y, min.z..=max.z) {
      let pos = Vector3::from([x, y, z]);
      let rel_chunk = Self::pos_to_chunk_id(pos) - cid_min;
      let chunk_offset = pos.map(|v| v.rem_euclid(CHUNK_SIZE_I32)); //cgmath doesn't (yet) have rem_euclid built in, which is different to normal rem and useful for negative numbners.
      
      if pos != (rel_chunk+cid_min)*CHUNK_SIZE_I32+chunk_offset {
        panic!("Bad position: {pos:?}, cid_min: {cid_min:?}, rel_chunk: {rel_chunk:?}, chunk_offset: {chunk_offset:?}");
      }
      //Get the chunk vis data from the `chunk_vis` vec.
      let chunk_data = chunk_vis.get(
        (rel_chunk.z + //i made my variable names too long help
        rel_chunk.y * cid_len.z +
        rel_chunk.x * (cid_len.y * cid_len.z)) as usize
      ).unwrap();

      //Get visibility data.
      let vis = chunk_data.get_block_at(chunk_offset);

      vis_data.push(vis);
    }
    
    BlockVisArea {
      data: vis_data,
      size: (max - min).map(|int| int as usize + 1) //Add 1 because it is inclusive.
    }
  }

  fn reuse_column(&self, column: &mut ChunkColumn, new_bounds: [[i32; 3]; 2], column_pos: [i32; 2], regen: [ChunkRegenCoord; 3]) {
    let [ncx, ncz] = column_pos;
    //Variable names for simplicity
    let old_start = self.chunk_id_bounds[0][1];
    let old_end = self.chunk_id_bounds[1][1];
    let new_start = new_bounds[0][1];
    let new_end = new_bounds[1][1];

    let y_range_size = new_end - new_start;
    let old_chunk_list = mem::replace(&mut column.chunks, Vec::<Arc<Chunk>>::with_capacity(y_range_size as usize));
    
    if new_start < old_end || new_end > old_start { //If there is an overlap.
      //Varaible names correspond to arraystuff.png
      let red = (new_start - old_start).max(0);
      let green = old_start.max(new_start);
      let blue = old_end.min(new_end);

      let mut useful_old_chunks = old_chunk_list.into_iter();
      for _ in 0..red {
        let chunk = useful_old_chunks.next().unwrap();
        self.chunk_gc.send(chunk).unwrap();
      }

      for ncy in new_start..green {
        let new_chunk = make_new_chunk([ncx, ncy, ncz]);
        column.chunks.push(new_chunk)
      }

      for ncy in green..blue {
        let reused_chunk = useful_old_chunks.next();
        match reused_chunk {
          Some(reused_chunk) => {
            let needs_regen = [ncx, ncy, ncz].iter().zip(regen.iter()).any(|(val, regen_coord)| {
              match regen_coord { //Check if the chunk needs regenerating.
                ChunkRegenCoord::None => false,
                ChunkRegenCoord::One(new_val) => val == new_val,
                ChunkRegenCoord::Both(lo, hi) => val == lo || val == hi,
              }
            });
            if needs_regen {
              reused_chunk.mark_for_revis();
            }
            column.chunks.push(reused_chunk);
          },
          None => {
            panic!("Reused chunk was none.")
          },
        }
      }

      for ncy in blue..new_end {
        let new_chunk = make_new_chunk([ncx, ncy, ncz]);
        column.chunks.push(new_chunk)
      }

      useful_old_chunks.for_each(|remaining_chunk| { //Send remaining chunks to gc.
        self.chunk_gc.send(remaining_chunk).unwrap();
      });

    } else { //Completely new chunks.
      for ncy in new_start..new_end {
        let new_chunk = make_new_chunk([ncx, ncy, ncz]);
        column.chunks.push(new_chunk)
      }

      for chunk in old_chunk_list {
        self.chunk_gc.send(chunk).unwrap();
      }
    }
  }

  pub fn pos_to_chunk_id(pos: Vector3<i32>) -> Vector3<i32> {
    pos.map(|v| if v < 0 {(v+1)/CHUNK_SIZE_I32 - 1} else {v / CHUNK_SIZE_I32})
  }
}

fn make_new_chunk(chunk_id: [i32; 3]) -> Arc<Chunk> {
  let new_chunk = Chunk::new(chunk_id);
  Arc::new(new_chunk)
}

/// A column of chunks. Includes the heightmap for the chunk.
struct ChunkColumn {
  pub chunks: Vec<Arc<Chunk>>,
  pub height_map: Arc<SurfaceHeightmap>
}

impl ChunkColumn {
  fn new(gen: &Perlin, chunk_xz: [i32; 2]) -> Self {
    let noise_coords = chunk_xz.map(|val| (val*CHUNK_SIZE_I32) as f64);
    
    let mut height_map: SurfaceHeightmap = [0i32; HEIGHTMAP_SIZE];
    for ((x,z), hm) in iproduct!(CHUNK_RANGE, CHUNK_RANGE).zip(height_map.iter_mut()) {
      let minor_hm = gen.get([
        (noise_coords[0] + x as f64 + 18284.0) / 30.0,
        (noise_coords[1] + z as f64 - 54761.0) / 30.0
      ]) * 5.0 + 5.0;

      let major_hm = gen.get([
        (noise_coords[0] + x as f64 - 4892.0) / 300.0,
        (noise_coords[1] + z as f64 + 645456.0) / 300.0
      ]) * 50.0 + 50.0;

      *hm = (major_hm + minor_hm + 5.0) as i32;
    }

    Self {
      chunks: Vec::new(),
      height_map: Arc::new(height_map)
    }
  }
}

impl BlockVisArea {
  fn get_block_at(&self, pos: Vector3<usize>) -> BlockSideVisibility {
    //Bounds check to prevent weird bugs.
    self.size.zip(pos, |s, p| if p >= s {
      panic!("Invalid block position. Pos: {0:?}. Size: {1:?}.", pos, self.size);
    });

    *self.data.get(pos.x * self.size.y * self.size.z + pos.y * self.size.z + pos.z).unwrap()
  }
}

#[derive(Clone, Copy, Debug)]
enum ChunkRegenCoord {
  None,
  One(i32),
  Both(i32, i32)
}


//Create an iterator of ints between 2 float points.
fn get_layers_between(a: Fixed64, b: Fixed64, inclusive: bool) -> Option<RangeStepInclusive<i32>> {
  let is_reversed = a>b;
  let (a_round, b_round) = if is_reversed ^ /* XOR */ inclusive {( //Is reversed. a is greater.
    a.floor().to_num::<i32>(),
    b.ceil().to_num::<i32>()
  )} else {( //Not reversed. b is greater.
    a.ceil().to_num(),
    b.floor().to_num()
  )};

  //TODO fix bug here.
  if (!is_reversed && a_round>b_round) || (is_reversed && b_round>a_round) || (a==b) { //No blocks checked.
    None
  } else {
    Some(range_step_inclusive(a_round, b_round, if is_reversed {-1} else {1})) //Cast to f32 for further calculation
  }
}


//Tests to make sure get_layers_between(..) works properly.
#[cfg(test)]
mod tests {
  use itertools::Itertools;

use crate::util::Fixed64;

  use super::get_layers_between;

  fn test_range(a: f32, b: f32, incl: bool, expt: Option<Vec<i32>>) {
    let lb = get_layers_between(Fixed64::from_num(a), Fixed64::from_num(b), incl);
    assert_eq!(lb.is_some(), expt.is_some());

    if expt.is_none() {
      return;
    }

    let lb_vec = lb.unwrap().collect_vec();
    let expt_vec = expt.unwrap();
    assert_eq!(lb_vec, expt_vec);
  }

  #[test]
  fn test_layers_pos() {
    test_range(0.5, 2.5, false, Some(vec![1, 2]));
  }

  #[test]
  fn test_layers_neg() {
    test_range(2.5, 0.5, false, Some(vec![2, 1]));
  }

  #[test]
  fn test_empty() {
    test_range(0.1, 0.5, false, None);
  }

  #[test]
  fn test_layers_val_neg() {
    test_range(0.1, -1.5, false, Some(vec![0, -1]));
  }
}