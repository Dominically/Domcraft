//i stole this

//sue me if bad
//https://docs.microsoft.com/en-us/windows/win32/direct3d9/projection-transform

use std::ops::{Add, AddAssign, Sub, SubAssign, Mul};

use cgmath::{Rad, Matrix4, Zero, Vector3};
use fixed::types::I32F32;

pub type Fixed64 = I32F32;

#[derive(Clone, Copy, Debug)]
pub struct FPVector {
  pub inner: Vector3<Fixed64>
}

impl FPVector {
  pub fn get_int(self) -> Vector3<i32> {
    self.inner.map(|v| v.int().to_num())
  }

  pub fn get_dec(self) -> Vector3<f32> {
    self.inner.map(|v| v.frac().to_num()) //TODO test
  }
}

impl Add<FPVector> for FPVector {
  type Output = FPVector;

  fn add(self, rhs: FPVector) -> Self::Output {
    Self {
      inner: self.inner.zip(rhs.inner, |l, r| l + r)
    }
  }
}

impl AddAssign<FPVector> for FPVector {
  fn add_assign(&mut self, rhs: FPVector) {
    *self = *self + rhs;
  }
}

impl Sub<FPVector> for FPVector {
  type Output = FPVector;

  fn sub(self, rhs: FPVector) -> Self::Output {
      Self {
        inner: self.inner.zip(rhs.inner, |l, r| l - r)
      }
  }
}

impl SubAssign<FPVector> for FPVector {
  fn sub_assign(&mut self, rhs: FPVector) {
    *self = *self - rhs;
  }
}

//TODO optimise this to be less repetitive.

impl From<Vector3<f32>> for FPVector {
  fn from(value: Vector3<f32>) -> Self {
    Self {
      inner: value.map(I32F32::from_num)
    }
  }
}

impl From<Vector3<i32>> for FPVector {
  fn from(value: Vector3<i32>) -> Self {
    Self {
      inner: value.map(I32F32::from_num)
    }
  }
}

impl From<[f32; 3]> for FPVector{
  fn from(value: [f32; 3]) -> FPVector {
    Self {
      inner: Vector3::from(value).map(I32F32::from_num)
    }
  }
}

impl From<[i32; 3]> for FPVector{
  fn from(value: [i32; 3]) -> FPVector {
    Self {
      inner: Vector3::from(value).map(I32F32::from_num)
    }
  }
}

impl Mul<Fixed64> for FPVector {
  type Output = FPVector;

  fn mul(self, rhs: Fixed64) -> Self::Output {
    Self {
      inner: self.inner.map(|v| {v * rhs})
    }
  }
}


pub fn projection<R: Into<Rad<f32>>>(fovy: R, aspect: f32, near_plane: f32, far_plane: f32) -> Matrix4<f32> {
  let fovy: Rad<f32> = fovy.into();
  let fovy = fovy.0/2.0;

  let h = 1.0/fovy.tan();
  let w = h/aspect;
  let q = far_plane/(far_plane - near_plane);

  let mut result = Matrix4::<f32>::zero();

  result.x.x = w;
  result.y.y = h;
  result.z.z = q;
  result.w.z = -q*near_plane;
  result.z.w = 1.0;

  result
}