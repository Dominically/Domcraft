use std::f32::consts::PI;

use cgmath::{Matrix4, Rad, Deg, Matrix3, Point3, num_traits::clamp};

use crate::stolen::projection;

const DEFAULT_FOV:f32 = 75.0;

pub struct Player {
  pub position: Point3<f32>,
  yaw: Rad<f32>,
  pitch: Rad<f32>,
  pub fov: f32
}

impl Player {
  pub fn new(position: Point3<f32>) -> Self {
    Self {
      position,
      yaw: Rad(0.0),
      pitch: Rad(0.0),
      fov: DEFAULT_FOV
    }
  }

  pub fn get_view_matrix(&self, aspect_ratio: f32) -> Matrix4<f32> {
    let rotation = self.get_rotation_matrix();
    let view = Matrix4::look_to_lh(self.position, rotation.z, rotation.y);
    let projection = projection(Deg(self.fov), aspect_ratio, 0.1, 400.0);

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
}