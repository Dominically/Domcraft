use std::{ops::Range, sync::{Arc, mpsc::Sender}, mem, cmp::Ordering};

use cgmath::{Vector3, Point3, Bounded, EuclideanSpace};
use itertools::iproduct;
use noise::{Perlin, NoiseFn, Seedable};

use crate::world::chunk::ChunkDataView;

use super::{chunk::{Chunk, ChunkMeshData, ChunkStateStage, ADJACENT_OFFSETS}, player::{PlayerPosition, HitBox}, chunk_worker_pool::{ChunkTask, ChunkTaskType}, block::{Block, BlockSide, BlockSideVisibility}};

pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_SIZE_I32: i32 = CHUNK_SIZE as i32;
pub const HEIGHTMAP_SIZE: usize = CHUNK_SIZE*CHUNK_SIZE;
pub const CHUNK_LENGTH: usize = HEIGHTMAP_SIZE*CHUNK_SIZE;
pub const CHUNK_RANGE: Range<usize> = 0..CHUNK_SIZE;

pub type SurfaceHeightmap = [i32; HEIGHTMAP_SIZE];

pub struct ChunkedTerrain {
  columns: Vec<ChunkColumn>, //Sorted in x ascending, then z ascending,
  chunk_id_bounds: [[i32; 3]; 2],
  player_last_chunk_id: [i32; 3], //The last Chunk ID of the player.
  render_distance: u32,
  worker_pool_sender: Sender<ChunkTask>,
  gen: Arc<Perlin>,
  chunk_gc: Sender<Arc<Chunk>>
}

impl ChunkedTerrain {
  pub fn new(player_position: PlayerPosition, render_distance: u32, worker_pool_sender: Sender<ChunkTask>, chunk_gc: Sender<Arc<Chunk>>) -> Self {
    let player_chunk_id = player_position.block_int.map(|val| val/CHUNK_SIZE_I32);
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
  pub fn update_player_position(&mut self, player_position: &PlayerPosition) -> bool { //Similar to the new() function but uses existing chunks if necessary.
    let player_chunk_id: [i32; 3] = player_position.block_int.map(|val| val / CHUNK_SIZE_I32).into();
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
  pub fn get_collision_info(&self, old_pos: PlayerPosition, delta: Vector3<f32>, hitbox: &HitBox) -> PlayerPosition {
    let target_pos = old_pos + delta; //The position if there is no collision.
    
    let hitbox_size = hitbox.hi - hitbox.lo;

    //Calculate the absolute positions of the hitbox corners.
    let a1 = old_pos + hitbox.lo;
    let b1 = old_pos + hitbox.hi;
    let a2 = target_pos + hitbox.lo;
    let b2 = target_pos + hitbox.hi;

    //Work out bounds of hitbox check. This is so we can work in relative coordinates to preserve floating-point accuracy.
    let (min, max) = [a1, b1, a2, b2].into_iter().fold((a1.block_int, b1.block_int), |(min, max), pos| ( //Start with default positions.
      min.zip(pos.block_int, |p, q| p.min(q)),
      max.zip(pos.block_int, |p, q| p.max(q))
    ));

    let (min, max) = (min.to_vec(), max.to_vec()); //Cast min and max to vectors. This should be a zero cost type cast.

    //Calculate player positions relative to the lesser corner of the hitbox check area.
    //Don't know if I need this yet.
    let rel_old_pos = old_pos - min;
    let rel_new_pos = target_pos - min;

    //Get terrain information.
    
    for (x, y, z) in iproduct!(min.x..max.x, min.y..max.y, min.z..max.z) {
      let pos = Vector3{x, y, z};
      
      
      
    }

    
    //Contains info about hitbox faces. `false` = select from lesser corner, `true` = select from greater corner.
    // const FACE_VERT_INFO: [[[bool; 3]; 2]; 6] = [
    //   [[true, true, true], [true, false, false]], //Right facing side (Pos X).
    //   [[false, true, true], [false, false, false]], //Left facing side (Neg X).
    //   [[true, true, true], [false, true, false]], //Upward facing side (Pos Y).
    //   [[true, false, true], [false, false, false]], //Downward facing side (Neg Y).
    //   [[true, true, true], [false, false, true]], //Backward facing side (Pos Z).
    //   [[true, true, false], [false, false, false]] //Forward facing side (Neg Z).
    // ];

    let sides_to_test = [ //Determine the faces which we need to test collision for (the direction the player is heading in).
      if delta.x >= 0.0 {0usize} else {1},
      if delta.y >= 0.0 {2} else {3},
      if delta.z >= 0.0 {4} else {5}
    ];

    for side in sides_to_test { //Now test sides for collision.
      let is_positive_dir = side%2==0; //true = hi, false = lo
      let dir_dim = side/3; //0 = x, 1 = y, 2 = z

      let [(a_old, b_old), (a_new, b_new)] = [old_pos, target_pos].map(|p|(
        p - min + if is_positive_dir {hitbox.hi} else {hitbox.lo},
        p - min + Vector3::from([0usize,1,2].map(|i| if is_positive_dir ^ (i==dir_dim) {hitbox.lo[i]} else {hitbox.hi[i]})) //Use coordinates of opposite corner except for the dimension of the side we are testing.
      ));

      //TODO continue from here.
    }

    
    
    
    return target_pos; //TODO temp for now.
    //This code is commented out because I don't think it's gonna work.

    // //Calculate absolute block positions of hitbox.
    // let ba = (old_pos + hitbox.a).block_int;
    // let bb = (old_pos + hitbox.b).block_int;

    // let test_surfaces = [
    //   [ba, Point3::from([bb.x, bb.y, ba.z])],
    //   [ba, Point3::from([bb.x, ba.y, bb.z])],
    //   [ba, Point3::from([ba.x, bb.y, bb.z])],
    //   [bb, Point3::from([ba.x, ba.y, bb.z])],
    //   [bb, Point3::from([ba.x, bb.y, ba.z])],
    //   [bb, Point3::from([bb.x, ba.y, ba.z])]
    // ];

    // test_surfaces.map(|[p, q]| { //Test from point p to point q and find collisions.
    //   let flagged = false;
    //   for (x, y, z) in iproduct!(p.x..q.x, p.y..q.y, p.z..q.z) {
    //     let block = self.get_block_at(x, y, z);
    //     match block {
    //       Some(Block::Air) => {}, //Do nothign
    //       _ =>
    //     }
    //   }
      
    //   flagged //Return whether surface has collision.
    // })
  }

  
  //Get visibility of blocks in a given area. This is used for hitbox testing.
  //TODO implement this so that the borders of chunks are solid to prevent the player from passing through bad chunks.
  fn get_block_vis_area(&self, min: Vector3<i32>, max: Vector3<i32>) -> Vec<BlockSideVisibility> {
    //Input validation check (this should never panic ideally).
    min.zip(max, |mn, mx| if mn > mx {panic!("Invalid range passed to get_block_vis_area.")});


    //Get positions of chunks we need.
    let [cid_min, cid_max] = [min, max].map(|v| v / CHUNK_SIZE_I32);
    //Create a range from positions.
    let cid_len = cid_min.zip(cid_max, |mn, mx| mx - mn);

    let mut vis_data = Vec::<BlockSideVisibility>::with_capacity(((max.x-min.x)*(max.y-min.y)*(max.z-min.z)) as usize); 
    
    let mut chunk_vis = Vec::new(); //probably not needed to use with_capacity as it won't usually be that many chunks

    for (cx, cy, cz) in iproduct!(cid_min.x..cid_max.x, cid_min.y..cid_max.y, cid_min.z..cid_max.z) {
      let data = self.get_chunk_at(&[cx, cy, cz]).map_or_else(|| ChunkDataView::new_blank(), |c| c.get_data_view());
      chunk_vis.push(data);
    }

    for (x, y, z) in iproduct!(min.x..max.x, min.y..max.y, min.z..max.z) {
      let rel_pos = Vector3::from([x, y, z]) - min; //Get relative position to the minimum.
      let rel_pos_chunk = rel_pos / CHUNK_SIZE_I32;
      let rel_chunk_offset = rel_pos.map(|v| v.rem_euclid(CHUNK_SIZE_I32)); //cgmath doesn't (yet) have rem_euclid built in, which is different to normal rem and useful for negative numbners.

      //Get the chunk vis data from the `chunk_vis` vec.
      let chunk_data = chunk_vis.get(
        (rel_pos_chunk.z + //i made my variable names too long help
        rel_pos_chunk.y * cid_len.z +
        rel_pos_chunk.x * (cid_len.y * cid_len.z)) as usize
      ).unwrap();

      //Get visibility data.
      let vis = chunk_data.get_block_at(rel_chunk_offset);

      vis_data.push(vis);
    }
    
    vis_data
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
      //Varaible names correspond to arrayshit.png
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

#[derive(Clone, Copy, Debug)]
enum ChunkRegenCoord {
  None,
  One(i32),
  Both(i32, i32)
}
 