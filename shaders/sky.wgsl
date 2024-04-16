struct VertexInput {
  @location(0) screen_position: vec2<f32>,
}

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) direction: vec4<f32>, //idk if this is right i will sort this later
}

struct SkyCameraUniform {
  view_matrix_inv: mat4x4<f32>
}

struct SkyFragmentUniform {
  sun_direction: vec3<f32>,
  light_level: f32,
}

@group(0) @binding(0)
var<uniform> camera: SkyCameraUniform;


@group(1) @binding(0)
var<uniform> data: SkyFragmentUniform;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
  var out: VertexOutput;

  out.clip_position = vec4<f32>(in.screen_position, 0.0, 1.0);
  out.direction = camera.view_matrix_inv * vec4<f32>(in.screen_position, 1.0, 1.0); //reminder: cosine similarity is NOT affected by the magnitudes of the vectors
  // out.colour = vec4<f32>(vec2<f32>(0.5, 0.5) + in.screen_position/vec2<f32>(2.0, 2.0), 0.0, 1.0);
  return out;
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let sky_colour = vec4<f32>(0.1, 0.24, 1.0, 1.0);
  let sun_colour = vec4<f32>(1.0, 1.0, 1.0, 1.0);
  
  let cosine_similarity = dot(normalize(in.direction.xyz), data.sun_direction);
  let sun_intensity = pow(clamp(cosine_similarity, 0.0, 1.0), 2048.0)*1.5;
  return (sun_colour * sun_intensity) + (sky_colour * data.light_level * (1.0 - sun_intensity));
}
