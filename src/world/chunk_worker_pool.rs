use std::sync::{mpsc::Receiver, Arc, RwLock, Mutex};

use wgpu::{Queue, Device, BindGroupLayout};

use super::chunk::Chunk;


pub type ChunkType = Arc<RwLock<Chunk>>;

pub fn run_worker_pool(device: Arc<Device>, queue: Arc<Queue>, chunk_reciever: Arc<Mutex<Receiver<ChunkType>>>, thread_id: usize) {
  'thread_loop: loop {
    let recieved = chunk_reciever.lock().unwrap().recv();
    match recieved{
        Ok(chunk) => {
          chunk.read().unwrap().update_vertices(&device, &queue);
        },
        Err(_) => break 'thread_loop,
    }
  }
}