use std::{sync::{mpsc::{Receiver, Sender, channel}, Arc, RwLock, Mutex}, thread::{JoinHandle, spawn}};

use crate::world::chunk::{Chunk, ChunkVertex};

type ChunkType = Arc<RwLock<Chunk>>;

type ChunkTask = ChunkType; //At the moment its just a simple job.

type ChunkResult = Vec<ChunkVertex>;

//TODO Write a trait for jobs.

///Starts a worker thread to recieve jobs. Will exit when reciever fails.
fn worker_thread(job_reciever: Arc<Mutex<Receiver<ChunkTask>>>, job_results: Sender<ChunkResult>) {
  'worker: loop {
    let job = job_reciever.lock().unwrap().recv();
    match job {
      Ok(job) => {
        //Run task.
        let verts = job.read().unwrap().get_vertices();
        job_results.send(verts); //Send results of job.

      },
      Err(_) => break 'worker, //Break the loop if no more tasks are reciever
    }
  }
}

pub struct ChunkWorkerPool {
  thread_handles: Vec<JoinHandle<()>>,
  job_sender: Sender<ChunkTask>
}

impl ChunkWorkerPool {
  pub fn new(result_output: Sender<ChunkResult>) -> Self {
    //Make one worker per CPU thread.
    let cpus = num_cpus::get();

    let (tx, rx) = channel();

    
    let arc_rx = Arc::new(Mutex::new(rx));

    let mut thread_handles = Vec::with_capacity(cpus);

    for _ in 0..cpus {
      let job_results = result_output.clone();
      let job_reciever = arc_rx.clone();
      let thread_handle = spawn(move || {
        worker_thread(job_reciever, job_results);
      });
      thread_handles.push(thread_handle);
    }

    Self {
      job_sender: tx,
      thread_handles
    }
  }
}
