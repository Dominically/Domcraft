use std::{borrow::Borrow, sync::{mpsc::{channel, Receiver}, Arc, Mutex}, thread, time::{Duration, Instant}};

use cgmath::Vector3;
use wgpu::Device;
use winit::{dpi::PhysicalPosition, event::{DeviceEvent, ElementState, Event, VirtualKeyCode, WindowEvent}, event_loop::{ControlFlow, EventLoop}, window::WindowBuilder};
use world::{chunk_worker_pool, chunk::Chunk};

use crate::{renderer::Renderer, world::World};

mod renderer;
mod world;
mod util;

pub type ArcWorld = Arc<Mutex<World>>;

fn main() {
  println!("Starting Domcraft...");
  pollster::block_on(run());
  // do_thing();
}

type FixedPoint = fixed::FixedI64::<fixed::types::extra::U32>;
fn fixed(val: f32) -> FixedPoint{
  FixedPoint::from_num(val)
}

fn do_thing() {
  let a = Vector3::from([12.5, 0.0, -5.2].map(fixed));
  let b = Vector3::from([1.2, 2.3, 3.4].map(fixed));
  let floored: Vector3<i32> = a.map(|v| v.to_num());
  println!("Test: {:?}", a);
  println!("Floored: {:?}", floored);
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

  window.set_cursor_visible(false);

  let mut is_focused = true;

  window.focus_window();
  event_loop.run(move |evt, _, ctrl| {
    match &evt {
      Event::WindowEvent { window_id, event } if window.id() == *window_id => {
        match event {
          WindowEvent::CloseRequested => {
            *ctrl = ControlFlow::Exit;
          },
          WindowEvent::Resized(new_size) => {
            renderer.resize(*new_size);
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

                window.set_cursor_visible(world.is_mouse_unlocked());
              }
              _ => ()
            }
          },
          WindowEvent::Focused(f) => {
            is_focused = *f;
          }
          _ => ()
        }
      }
      Event::RedrawRequested( window_id ) if window.id() == *window_id => {
        renderer.render().unwrap();        
      },
      Event::DeviceEvent { device_id: _, event } if is_focused => { //Raw input from val?
        match event {
          winit::event::DeviceEvent::MouseMotion { delta } => {
            let mut world = world.lock().unwrap();
            if !world.is_mouse_unlocked() {
              world.mouse_move(*delta);
              let _ = window.set_cursor_position(PhysicalPosition::new(
                window.inner_size().width/2,
                window.inner_size().height/2
              ));
            } else {
              
            }
          },
          _ => ()
        }
      },
      Event::MainEventsCleared => {
        window.request_redraw()
      }
      _ => (),
    };

    let mut default_handle_evt = |bypass_check: bool| { //idk why this has to be mut
      if bypass_check || world.lock().unwrap().is_mouse_unlocked() {
        renderer.imgui().window_event(&window, &evt);
      }
    };

    match &evt {
      Event::DeviceEvent { device_id: _, event: device_event } => {
        if let DeviceEvent::Key(key_event) = device_event {
          default_handle_evt(key_event.state == ElementState::Released); //Bypass mouse check if key is released.
        } else {
          default_handle_evt(false);
        }
      }
      _ => {
        default_handle_evt(true);
      }
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
