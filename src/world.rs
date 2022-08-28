use std::time::Instant;

use cgmath::{Matrix4, Rad, Vector3};
use winit::event::VirtualKeyCode;

use self::{player::Player, controls::{Controller, Control}, chunkedterrain::ChunkedTerrain};

mod block;
mod player;
mod controls;
pub mod chunkedterrain;
pub mod chunk;


const MOUSE_SENS: Rad<f32> = Rad(0.002); //Rads per dot.
const NOCLIP_SPEED: f32 = 5.0; //blocks/sec
pub struct World {
  terrain: ChunkedTerrain,
  player: Player,
  last_tick: Instant,
  controller: Controller
}


impl World {
  pub fn new() -> Self {
    let player_pos: [f32; 3] = [128.0, 40.0, 128.0];
    
    let player = Player::new(player_pos.into());
    
    let terrain = ChunkedTerrain::new(player.position.into() as [f64; 3], 64);
    let last_tick = Instant::now();
    let mut controller = Controller::new();

    controller.set_bindings(&[
      (VirtualKeyCode::W, Control::Forward),
      (VirtualKeyCode::A, Control::Left),
      (VirtualKeyCode::S, Control::Backward),
      (VirtualKeyCode::D, Control::Right),
      (VirtualKeyCode::Space, Control::Up),
      (VirtualKeyCode::LControl, Control::Down),
      (VirtualKeyCode::Up, Control::Forward),
      (VirtualKeyCode::Left, Control::Left),
      (VirtualKeyCode::Down, Control::Backward),
      (VirtualKeyCode::Right, Control::Right),
      (VirtualKeyCode::RShift, Control::Up),
      (VirtualKeyCode::RControl, Control::Down)
    ]);
    
    Self {
      terrain,
      player,
      last_tick,
      controller
    }
  }

  pub fn get_player_view(&self, aspect_ratio: f32) -> Matrix4<f32> {
    self.player.get_view_matrix(aspect_ratio)
  }

  pub fn mouse_move(&mut self, delta: (f64, f64)) {
    self.player.rotate_camera(MOUSE_SENS * delta.0 as f32, MOUSE_SENS * delta.1 as f32);
  }

  pub fn tick(&mut self) {
    let now = Instant::now();
    let delta_secs = (now - self.last_tick).as_secs_f32();
    self.last_tick = now;

    let x_speed = self.controller.get_action_value((Control::Left, -1.0), (Control::Right, 1.0), 0.0);
    let y_speed = self.controller.get_action_value((Control::Down, -1.0), (Control::Up, 1.0), 0.0);
    let z_speed = self.controller.get_action_value((Control::Backward, -1.0), (Control::Forward, 1.0), 0.0);

    let direction_vector = Vector3 {
      x: x_speed,
      y: y_speed,
      z: z_speed
    };

    let displacement = self.player.get_rotation_matrix() * direction_vector * NOCLIP_SPEED * delta_secs;
    self.player.position += displacement;
  }

  pub fn key_update(&mut self, key: VirtualKeyCode, state: bool) {
    self.controller.set_key(key, state);
  }
}