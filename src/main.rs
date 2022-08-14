use winit::{window::{WindowBuilder}, event_loop::{EventLoop, ControlFlow}, event::{Event, WindowEvent}};

use crate::renderer::Renderer;

mod renderer;
mod world;

fn main() {
  println!("Hello, world!");
  pollster::block_on(run());
}

async fn run() {

  let event_loop = EventLoop::new();
  let window = WindowBuilder::new()
    .with_title("DomCraft [INDEV]").build(&event_loop).expect("Failed to create window!");
  
  let mut renderer = Renderer::new(&window).await.unwrap();

  event_loop.run(move |evt, _, ctrl| {
    *ctrl = ControlFlow::Wait;

    match evt {
      Event::WindowEvent { window_id, event } if window.id() == window_id => {
        match event {
          WindowEvent::CloseRequested => {
            *ctrl = ControlFlow::Exit;
          },
          WindowEvent::Resized(new_size) => {
            renderer.resize(new_size);
            renderer.render().unwrap();
          }
          _ => ()
        }
      }
      _ => (),
    }
  });


}
