use std::{f32::consts::PI, ops::{Add, AddAssign, SubAssign, Sub}, time::Duration};

use cgmath::{Matrix4, Rad, Deg, Matrix3, Point3, num_traits::clamp, Vector3};

use crate::stolen::projection;

use super::chunkedterrain::ChunkedTerrain;

const SPEED_FACTOR: f32 = 0.5;
const DEFAULT_FOV: f32 = 75.0;

const DEFAULT_HITBOX: HitBox = HitBox {
  lo: Vector3 {x: -0.5, y:-1.5, z: -0.5},
  hi: Vector3 {x: 0.5, y: 0.5, z: 0.5}
};

//TODO possibly split hitbox and position data into a separate entity data structure.

pub struct Player {
  position: PlayerPosition,
  velocity: Vector3<f32>,
  yaw: Rad<f32>,
  pitch: Rad<f32>,
  pub fov: f32,
  hitbox: HitBox
}

/**
 The hitbox is a 3-dimensional cuboid aligned with the world that cannot rotate. It is relevant for player physics.
 */
pub struct HitBox {
  pub lo: Vector3<f32>,
  pub hi: Vector3<f32>
}

/// The PlayerPosition is a fixed point integer because it is useful to not use accuracy at large distances (unlike floats).
#[repr(C)] //Repr(c) because this is being sent to GPU.
#[derive(Debug, Clone, Copy)]
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
      fov: DEFAULT_FOV,
      hitbox: DEFAULT_HITBOX
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

  /**
  Update player position in world.
   - `target_vel` - Target velocity.
   - `dt` - Duration since last tick.
   - `terrain` - World terrain data. 
   */
  pub fn tick_position(&mut self, target_vel: &Vector3<f32>, dt: &Duration, terrain: &ChunkedTerrain) {
    let diff = target_vel - self.velocity;
    let secs = dt.as_secs_f32();
    let factor = secs/(secs + SPEED_FACTOR);
    self.velocity += diff * factor;
    let position_delta = self.velocity * secs;
    
    self.position = terrain.get_collision_info(self.position, position_delta, &self.hitbox);

  }
}

impl AddAssign<Vector3<f32>> for PlayerPosition {
  fn add_assign(&mut self, rhs: Vector3<f32>) {
      let added_float = self.block_dec + rhs;
      self.block_int = self.block_int.zip(added_float, |s, t| s + t.trunc() as i32 + (if t<0.0 {-1} else {0})); //Add integer components.
      self.block_dec = added_float.map(|v| if v<0.0 {v.fract()+1.0} else {v.fract()}); //Add decimal components.
  }
}

//TODO cleanup messy boilerplate code.
impl Add<Vector3<f32>> for PlayerPosition {
    type Output = PlayerPosition;

    fn add(mut self, rhs: Vector3<f32>) -> Self::Output {
      self += rhs; //Use the AddAssign trait implemented below.
      self
    }
}

impl SubAssign<Vector3<f32>> for PlayerPosition {
  fn sub_assign(&mut self, rhs: Vector3<f32>) {
      *self += -rhs;
  }
}

impl Sub<Vector3<f32>> for PlayerPosition {
  type Output = PlayerPosition;

  fn sub(self, rhs: Vector3<f32>) -> Self::Output {
      self + -rhs
  }
}

impl AddAssign<Vector3<i32>> for PlayerPosition {
  fn add_assign(&mut self, rhs: Vector3<i32>) {
    self.block_int += rhs;
  }
}

impl Add<Vector3<i32>> for PlayerPosition {
  type Output = PlayerPosition;

  fn add(mut self, rhs: Vector3<i32>) -> Self::Output {
    self += rhs; //Use AddAssign trait again.
    self
  }
}

impl SubAssign<Vector3<i32>> for PlayerPosition {
  fn sub_assign(&mut self, rhs: Vector3<i32>) {
      *self += -rhs;
  }
}

impl Sub<Vector3<i32>> for PlayerPosition {
  type Output = PlayerPosition;

  fn sub(self, rhs: Vector3<i32>) -> Self::Output {
      self + -rhs
  }
}

#[cfg(test)]
mod test {
    use cgmath::{Vector3, Point3, EuclideanSpace, InnerSpace};

    use super::PlayerPosition;

  #[test]
  fn test_pos_add() {
    let a = PlayerPosition {
      block_dec: [0.0f32, 0.0, 0.0].into(),
      block_int: [1i32, 1, 1].into()
    };

    let added = a + Vector3::from([1.5f32, 1.5, 1.5]);
    
    assert!((added.block_dec.to_vec() - Vector3{x:0.5f32,y:0.5,z:0.5}).magnitude() < 0.01);
    assert_eq!(added.block_int, Point3{x:2i32,y:2,z:2});


    let neg_add = a + Vector3::from([-0.5f32, -0.5, -0.5]);
    assert!((neg_add.block_dec.to_vec() - Vector3{x:0.5,y:0.5,z:0.5}).magnitude() < 0.01);
    assert_eq!(neg_add.block_int, Point3{x:0i32,y:0,z:0});
  }
}