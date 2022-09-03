use std::sync::{mpsc::Receiver, Arc, RwLock, Mutex};

use wgpu::{Queue, Device, BindGroupLayout};

use super::chunk::Chunk;


pub type ChunkType = Arc<RwLock<Chunk>>;

pub fn run_worker_pool(device: Arc<Device>, queue: Arc<Queue>, chunk_reciever: Arc<Mutex<Receiver<ChunkType>>>) {
  'thread_loop: loop {
    match chunk_reciever.lock().unwrap().recv() {
        Ok(chunk) => {
          chunk.read().unwrap().update_vertices(&device, &queue);
        },
        Err(_) => break 'thread_loop,
    }
  }
}