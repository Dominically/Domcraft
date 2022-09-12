use std::{f32::consts::PI, ops::{Add, AddAssign}, time::Duration};

use cgmath::{Matrix4, Rad, Deg, Matrix3, Point3, num_traits::clamp, Vector3};

use crate::stolen::projection;

const SPEED_FACTOR: f32 = 0.5;
const DEFAULT_FOV: f32 = 75.0;

pub struct Player {
  position: PlayerPosition,
  velocity: Vector3<f32>,
  yaw: Rad<f32>,
  pitch: Rad<f32>,
  pub fov: f32
}

/// The PlayerPosition is a fixed point integer because it is useful to not use accuracy at large distances (unlike floats).
#[repr(C)] //Repr(c) because this is being sent to GPU.
#[derive(Clone, Copy)]
pub struct PlayerPosition {
  pub block_int: Point3<i32>, //The block integer.
  pub block_dec: Point3<f32> //The decimal part.
}

impl Player {
  pub fn new(position: PlayerPosition) -> Self {
    Self {
      position,
      velocity: Vector3 { x: 0.0, y: 0.0, z: 0.0 },
      yaw: Rad(0.0),
      pitch: Rad(0.0),
      fov: DEFAULT_FOV
    }
  }

  ///Gets the player view matrix relative to the nearest block. Conversions on integers still need to be done on the GPU.
  pub fn get_view_matrix(&self, aspect_ratio: f32, dt: Duration) -> Matrix4<f32> {
    let rotation = self.get_rotation_matrix();
    let pos_offset = self.velocity * dt.as_secs_f32(); //To prevent stuttering and lagging on high Hz monitors.
    let view = Matrix4::look_to_lh(self.position.block_dec + pos_offset, rotation.z, rotation.y);
    let projection = projection(Deg(self.fov), aspect_ratio, 0.1, 400.0); //Very very far far plane.

    projection * view
  }

  pub fn rotate_camera(&mut self, dx: impl Into<Rad<f32>>, dy: impl Into<Rad<f32>>) {
    let dxr = dx.into();
    let dyr = dy.into();

    self.yaw = (self.yaw + dxr) % Rad(2.0 * PI);
    self.pitch = clamp(self.pitch + dyr, Rad(-PI/2.0), Rad(PI/2.0));
  }

  pub fn get_rotation_matrix(&self) -> Matrix3<f32> {
    Matrix3::from_angle_y(self.yaw) * Matrix3::from_angle_x(self.pitch)
  }

  pub fn get_position(&self) -> PlayerPosition {
    self.position.clone()
  }

  pub fn tick_position(&mut self, target_vel: &Vector3<f32>, dt: &Duration) {
    let diff = target_vel - self.velocity;
    let secs = dt.as_secs_f32();
    let factor = secs/(secs + SPEED_FACTOR);
    self.velocity += diff * factor;
    self.position += self.velocity * secs;

  }
}

impl Add<Vector3<f32>> for PlayerPosition {
    type Output = PlayerPosition;

    fn add(mut self, rhs: Vector3<f32>) -> Self::Output {
      self += rhs; //Use the AddAssign trait implemented below.
      self
    }
}

impl AddAssign<Vector3<f32>> for PlayerPosition {
    fn add_assign(&mut self, rhs: Vector3<f32>) {
        let added_float = self.block_dec + rhs;
        self.block_int = self.block_int.zip(added_float, |s, t| s + t.trunc() as i32); //Add integer components.
        self.block_dec = added_float.map(|v| v.fract()); //Add decimal components.
    }
}