use std::{sync::{Mutex, Arc, mpsc::{channel, Receiver}}, thread, time::{Instant, Duration}};

use winit::{window::{WindowBuilder}, event_loop::{EventLoop, ControlFlow}, event::{Event, WindowEvent, ElementState, VirtualKeyCode}, dpi::PhysicalPosition};
use world::{chunk_worker_pool, chunk::Chunk};

use crate::{renderer::Renderer, world::World};

mod renderer;
mod world;
mod stolen;

pub type ArcWorld = Arc<Mutex<World>>;

fn main() {
  println!("Hello, world!2");
  pollster::block_on(run());
}

async fn run() {
  let event_loop = EventLoop::new();
  let window = WindowBuilder::new()
    .with_title("DomCraft [INDEV]").build(&event_loop).expect("Failed to create window!");
  let mut renderer = Renderer::new(&window).await.unwrap();

  let (worker_tx, worker_rx) = channel();
  let rx_arc = Arc::new(Mutex::new(worker_rx));
  let (device, queue) = renderer.get_device_queue();
  let cpu_count = num_cpus::get();
  let worker_thread_count = if cpu_count <= 2 {
    1
  } else {
    cpu_count - 2
  };
  for i in 0..worker_thread_count { //Spawn worker threads.
    let (device, queue, rx_arc) = (
      device.clone(),
      queue.clone(),
      rx_arc.clone()
    );
    thread::Builder::new().name(format!("Worker #{}", i)).spawn(move || {
      chunk_worker_pool::run_worker_thread(device, queue, rx_arc)
    }).unwrap();
  }

  //Spawn chunk GC thread.
  let (gc_tx, gc_rx) = channel();
  thread::Builder::new().name("GC Thread".to_string()).spawn(move || gc_thread(gc_rx)).unwrap();

  let world = Arc::new(Mutex::new(World::new(worker_tx, gc_tx)));

  renderer.bind_world(world.clone());

  {
    let tick_world = world.clone();
    thread::Builder::new().name("World Tick".to_string()).spawn(move || {
      world_tick_thread(tick_world);
    }).unwrap();
  }
  println!("Done making world.");
  
  println!("Bound world");

  window.set_cursor_visible(false);

  let mut is_focused = true;
  window.focus_window();
  event_loop.run(move |evt, _, ctrl| {

    match evt {
      Event::WindowEvent { window_id, event } if window.id() == window_id => {
        match event {
          WindowEvent::CloseRequested => {
            *ctrl = ControlFlow::Exit;
          },
          WindowEvent::Resized(new_size) => {
            renderer.resize(new_size);
          },
          WindowEvent::KeyboardInput { device_id: _, input, is_synthetic: _ } => {
            match input.virtual_keycode {
              Some(key) => {
                let mut world = world.lock().unwrap();
                if let VirtualKeyCode::Escape = key {
                  *ctrl = ControlFlow::Exit;
                  return;
                }
                match input.state {
                  ElementState::Pressed => world.key_update(key, true),
                  ElementState::Released => world.key_update(key, false),
                }
              }
              _ => ()
            }
          },
          WindowEvent::Focused(f) => {
            is_focused = f;
          }
          _ => ()
        }
      }
      Event::RedrawRequested( window_id ) if window.id() == window_id => {
        renderer.render().unwrap();        
      },
      Event::DeviceEvent { device_id: _, event } if is_focused => { //Raw input from val?
        match event {
          winit::event::DeviceEvent::MouseMotion { delta } => {
            let mut world = world.lock().unwrap();
            world.mouse_move(delta);
            let _ = window.set_cursor_position(PhysicalPosition::new(
              window.inner_size().width/2,
              window.inner_size().height/2
            ));
          },
          _ => ()
        }
      },
      Event::MainEventsCleared => {
        window.request_redraw()
      }
      _ => (),
    }
  });
}

fn world_tick_thread(world: Arc<Mutex<World>>) {
  const TICK_DELAY: f32 = 1.0/64.0;
  let tick_duration = Duration::from_secs_f32(TICK_DELAY);
  
  loop {
    let start_time = Instant::now();
    world.lock().unwrap().tick();
    let time_duration = Instant::now() - start_time;
    if time_duration < tick_duration {
      let diff = tick_duration - time_duration;
      thread::sleep(diff);
    }
  }
}

fn gc_thread(rx: Receiver<Arc<Chunk>>) {
  'gc: loop {
    if rx.recv().is_err() {
      break 'gc;
    }
  }
}
