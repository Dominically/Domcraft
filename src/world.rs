use std::{time::Instant, sync::{mpsc::Sender, Arc}};

use cgmath::{Matrix4, Rad, Vector3};
use winit::event::VirtualKeyCode;

use self::{player::{Player, PlayerPosition}, controls::{Controller, Control}, chunkedterrain::ChunkedTerrain, chunk_worker_pool::ChunkTask, chunk::Chunk};

mod block;
mod player;
mod controls;
pub mod chunkedterrain;
pub mod chunk;
pub mod chunk_worker_pool;


const MOUSE_SENS: Rad<f32> = Rad(0.002); //Rads per dot.
const NOCLIP_SPEED: f32 = 40.0; //blocks/sec
pub struct World {
  terrain: ChunkedTerrain,
  player: Player,
  last_tick: Instant,
  controller: Controller
}


impl World {
  pub fn new(worker_pool_sender: Sender<ChunkTask>, chunk_gc: Sender<Arc<Chunk>>) -> Self {
    let player_pos = PlayerPosition {
        block_int: [1, 40, 1].into(),
        block_dec: [0.0; 3].into(),
    };
    
    let player = Player::new(player_pos.into());
    
    let terrain = ChunkedTerrain::new(player_pos, 8, worker_pool_sender, chunk_gc);
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
    let delta_t = Instant::now() - self.last_tick;
    self.player.get_view_matrix(aspect_ratio, delta_t)
  }

  pub fn get_player_pos(&self) -> PlayerPosition {
    self.player.get_position()
  }

  pub fn mouse_move(&mut self, delta: (f64, f64)) {
    self.player.rotate_camera(MOUSE_SENS * delta.0 as f32, MOUSE_SENS * delta.1 as f32);
  }

  pub fn tick(&mut self) {
    let now = Instant::now();
    let delta_secs = now - self.last_tick;
    self.last_tick = now;

    let x_speed = self.controller.get_action_value((Control::Left, -1.0), (Control::Right, 1.0), 0.0);
    let y_speed = self.controller.get_action_value((Control::Down, -1.0), (Control::Up, 1.0), 0.0);
    let z_speed = self.controller.get_action_value((Control::Backward, -1.0), (Control::Forward, 1.0), 0.0);

    let direction_vector = Vector3 {
      x: x_speed,
      y: y_speed,
      z: z_speed
    };

    let accel = self.player.get_rotation_matrix() * direction_vector * NOCLIP_SPEED ;
    self.player.tick_position(&accel, &delta_secs);
    self.terrain.update_player_position(&self.player.get_position());

    self.terrain.tick_progress();
  }


  pub fn key_update(&mut self, key: VirtualKeyCode, state: bool) {
    self.controller.set_key(key, state);
  }

  pub fn get_terrain(&self) -> &ChunkedTerrain {
    &self.terrain
  }
}