//i stole this

//sue me if bad
//https://docs.microsoft.com/en-us/windows/win32/direct3d9/projection-transform

use cgmath::{Rad, Matrix4, Zero};

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