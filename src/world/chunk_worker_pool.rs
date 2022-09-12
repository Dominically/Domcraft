use std::sync::{mpsc::Receiver, Arc, Mutex};

use noise::Perlin;
use wgpu::{Queue, Device};

use super::{chunk::Chunk, chunkedterrain::SurfaceHeightmap};

pub enum ChunkTaskType {
  GenTerrain(Arc<Perlin>, Arc<SurfaceHeightmap>),
  GenBlockVis([Option<Arc<Chunk>>; 6]),
  GenVertices
}

pub struct ChunkTask {
  pub chunk: Arc<Chunk>,
  pub typ: ChunkTaskType
}

pub fn run_worker_pool(device: Arc<Device>, queue: Arc<Queue>, chunk_reciever: Arc<Mutex<Receiver<ChunkTask>>>) {
  'thread_loop: loop {
    let recieved = chunk_reciever.lock().unwrap().recv();
    match recieved {
      Ok(task) => {
        match task.typ {
            ChunkTaskType::GenTerrain(gen, surface_heightmap) => task.chunk.gen(&gen, &surface_heightmap),
            ChunkTaskType::GenBlockVis(surrounding_chunks) => task.chunk.gen_block_vis(surrounding_chunks),
            ChunkTaskType::GenVertices => task.chunk.update_vertices(&device, &queue),
        }
      },
      Err(_) => break 'thread_loop,
    }
  }
}