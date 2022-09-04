use std::{ops::Range, sync::{RwLock, Arc, mpsc::Sender}, mem, cmp::Ordering};

use itertools::iproduct;
use noise::{Perlin, NoiseFn};

use super::{chunk::{Chunk, ChunkMeshData}, player::PlayerPosition, chunk_worker_pool::ChunkType};

pub const CHUNK_SIZE: usize = 16;
pub const HEIGHTMAP_SIZE: usize = CHUNK_SIZE*CHUNK_SIZE;
pub const CHUNK_LENGTH: usize = HEIGHTMAP_SIZE*CHUNK_SIZE;
pub const CHUNK_RANGE: Range<usize> = 0..CHUNK_SIZE;

pub type SurfaceHeightmap = [i32; HEIGHTMAP_SIZE];

pub struct ChunkedTerrain {
  columns: Vec<ChunkColumn>, //Sorted in x ascending, then z ascending,
  chunk_id_bounds: [[i32; 3]; 2],
  player_last_chunk_id: [i32; 3], //The last Chunk ID of the player.
  render_distance: u32,
  worker_pool_sender: Sender<ChunkType>,
  gen: Perlin
}

impl ChunkedTerrain {
  pub fn new(player_position: PlayerPosition, render_distance: u32, worker_pool_sender: Sender<ChunkType>) -> Self {
    let player_chunk_id = player_position.block_int.map(|val| val/CHUNK_SIZE as i32);
    let chunk_id_bounds: [[i32; 3]; 2] = [
      player_chunk_id.map(|chk| chk-render_distance as i32).into(),
      player_chunk_id.map(|chk| chk+render_distance as i32).into()
    ];

    let gen = Perlin::new();
    
    let columns: Vec<ChunkColumn> = iproduct!(
      chunk_id_bounds[0][0]..chunk_id_bounds[1][0], 
      chunk_id_bounds[0][2]..chunk_id_bounds[1][2])
      .map(|(cx, cz)| {
        let mut column = ChunkColumn::new(&gen, [cx, cz]);
        for cy in chunk_id_bounds[0][1]..chunk_id_bounds[1][1] { //Iterate vertically
          column.chunks.push(make_new_chunk(&gen, [cx, cy, cz], &column.height_map));
        }
        column
      }).collect();
    
    
    Self {
      columns,
      render_distance,
      worker_pool_sender,
      chunk_id_bounds,
      player_last_chunk_id: player_chunk_id.into(),
      gen
    }
  }

  //Returns true if the chunk vertices need to be regenerated.
  pub fn update_player_position(&mut self, player_position: &PlayerPosition) -> bool { //Similar to the new() function but uses existing chunks if necessary.
    let player_chunk_id: [i32; 3] = player_position.block_int.map(|val| val / CHUNK_SIZE as i32).into();
    if player_chunk_id == self.player_last_chunk_id { //Skip the update if the player has not moved far enough to update the world.
      return false;
    }

    let new_bounds: [[i32; 3]; 2] = [
      player_chunk_id.map(|chk| chk-self.render_distance as i32).into(),
      player_chunk_id.map(|chk| chk+self.render_distance as i32).into()
    ];

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
        next_old_column = old_column_iter.next();
      }

      let column = match &next_old_column {
        //Column already exists.
        Some(col) if [col.0.0, col.0.1] == [ncx, ncz] => {
          let mut column = next_old_column.unwrap().1;
          next_old_column = old_column_iter.next();
          self.reuse_column(&mut column, new_bounds, [ncx, ncz]);
          column
        },
        //Create new column.
        _ => {
          let mut column = ChunkColumn::new(&self.gen, [ncx, ncz]);
          for ncy in new_bounds[0][1]..new_bounds[1][1] {
            let chunk = make_new_chunk(&self.gen, [ncx, ncy, ncz], &column.height_map);
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
        let chunk = chunk.read().unwrap();
        if let Some(mesh_data) = chunk.get_mesh() {
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

  pub fn get_chunk_at(&self, chunk_id: &[i32; 3]) -> Option<&Arc<RwLock<Chunk>>> {
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

  pub fn gen_block_vis(&mut self) {
    //Generate block visibility.
    for col in self.columns.iter() {
      for chunk in col.chunks.iter() {
        let [x, y, z] = chunk.read().unwrap().get_id();
        let surrounding_chunk_locks = [
          [1i32, 0, 0],
          [-1, 0, 0],
          [0, 1, 0],
          [0, -1, 0],
          [0, 0, 1],
          [0, 0, -1]
        ].map(|[ox, oy, oz]| {
          let chunk_id = [x + ox, y + oy, z + oz];
          let chk = self.get_chunk_at(&chunk_id).map(|chk| chk.read().unwrap());
          if let Some(chk) = &chk {
            if chk.get_id() != chunk_id {
              panic!("Found wrong chunk. Expected: {:?} Found: {:?}", chunk_id, chk.get_id())
            }
          }
          chk
        });

        //TODO wait for .each to actually become usable.
        let surrounding_chunk_refs = [ //it works
          surrounding_chunk_locks[0].as_deref(),
          surrounding_chunk_locks[1].as_deref(),
          surrounding_chunk_locks[2].as_deref(),
          surrounding_chunk_locks[3].as_deref(),
          surrounding_chunk_locks[4].as_deref(),
          surrounding_chunk_locks[5].as_deref(),
        ];
        {
          let mut chunk_write_lock = chunk.write().unwrap();
          chunk_write_lock.gen_block_vis(surrounding_chunk_refs);
        }
      }
    }
  }

  //Call chunk updates.
  pub fn send_chunk_update(&self) {
    for col in self.columns.iter() {
      for chunk in col.chunks.iter() {
        if chunk.read().unwrap().needs_updating() { //Only update the chunk if it needs it.
          self.worker_pool_sender.send(chunk.clone()).unwrap();
        }
      }
    }
  }

  fn reuse_column(&self, column: &mut ChunkColumn, new_bounds: [[i32; 3]; 2], column_pos: [i32; 2]) {

    let [ncx, ncz] = column_pos;
    //Variable names for simplicity
    let old_start = self.chunk_id_bounds[0][1];
    let old_end = self.chunk_id_bounds[1][1];
    let new_start = new_bounds[0][1];
    let new_end = new_bounds[1][1];

    let y_range_size = new_end - new_start;
    
    let old_chunk_list = mem::replace(&mut column.chunks, Vec::<Arc<RwLock<Chunk>>>::with_capacity(y_range_size as usize));

    if new_start < old_end || new_end > old_start { //If there is an overlap.
      //Varaible names correspond to arrayshit.png
      let red = (new_start - old_start).max(0);
      let green = old_start.max(new_start);
      let blue = old_end.min(new_end);

      let mut useful_old_chunks = old_chunk_list.into_iter().skip(red as usize);

      for ncy in new_start..green {  
        let new_chunk = make_new_chunk(&self.gen, [ncx, ncy, ncz], &column.height_map);
        column.chunks.push(new_chunk)
      }

      for _ncy in green..blue {
        let reused_chunk = useful_old_chunks.next();
        match reused_chunk {
          Some(reused_chunk) => {
            column.chunks.push(reused_chunk)
          },
          None => {
            panic!("Reused chunk was none.")
          },
        }
        
      }

      for ncy in blue..new_end {
        let new_chunk = make_new_chunk(&self.gen, [ncx, ncy, ncz], &column.height_map);
        column.chunks.push(new_chunk)
      }

    } else { //Completely new chunks.
      for ncy in new_start..new_end {
        let new_chunk = make_new_chunk(&self.gen, [ncx, ncy, ncz], &column.height_map);
        column.chunks.push(new_chunk)
      }
    }
  }
}

fn make_new_chunk(gen: &Perlin, chunk_id: [i32; 3], height_map: &SurfaceHeightmap) -> Arc<RwLock<Chunk>> {
  let new_chunk = Chunk::new(gen, chunk_id, height_map);
  Arc::new(RwLock::new(new_chunk))
  
}

/// A column of chunks. Includes the heightmap for the chunk.
struct ChunkColumn {
  pub chunks: Vec<Arc<RwLock<Chunk>>>,
  pub height_map: SurfaceHeightmap
}

impl ChunkColumn {
  fn new(gen: &Perlin, chunk_xz: [i32; 2]) -> Self {
    let noise_coords = chunk_xz.map(|val| (val*CHUNK_SIZE as i32) as f64);
    
    let mut height_map: SurfaceHeightmap = [0i32; HEIGHTMAP_SIZE];
    for ((x,z), hm) in iproduct!(CHUNK_RANGE, CHUNK_RANGE).zip(height_map.iter_mut()) {
      *hm = (gen.get([
        (noise_coords[0] + x as f64) / 30.0,
        (noise_coords[1] + z as f64) / 30.0
      ]) * 5.0 + 10.0) as i32;
    }

    Self {
      chunks: Vec::new(),
      height_map
    }
  }
}
 