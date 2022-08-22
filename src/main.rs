use std::sync::{Mutex, Arc};

use winit::{window::{WindowBuilder}, event_loop::{EventLoop, ControlFlow}, event::{Event, WindowEvent, ElementState}, dpi::LogicalPosition};

use crate::{renderer::Renderer, world::World};

mod renderer;
mod world;
mod stolen;

pub type ArcWorld = Arc<Mutex<World>>;

fn main() {
  println!("Hello, world!");
  pollster::block_on(run());
}

async fn run() {
  let world = Arc::new(Mutex::new(World::new()));
  let event_loop = EventLoop::new();
  let window = WindowBuilder::new()
    .with_title("DomCraft [INDEV]").build(&event_loop).expect("Failed to create window!");
  let mut renderer = Renderer::new(&window, world.clone()).await.unwrap();
  
  window.set_cursor_visible(false);

  let mut is_focused = true;
  window.focus_window();
  event_loop.run(move |evt, _, ctrl| {
    let mut wrld = world.lock().unwrap();
    wrld.tick();
    drop(wrld);

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
            //let _ = window.set_cursor_position(LogicalPosition::new(0.5, 0.5));
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
