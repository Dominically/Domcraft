// Vertex shader
struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) colour: vec3<f32>,
};


struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) colour: vec3<f32>,
};

struct CameraUniform {
    view_proj: mat4x4<f32>
}

@group(0) @binding(0) // 1.
var<uniform> camera: CameraUniform;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4(in.position, 1.0);
    out.colour = pow(in.colour, vec3<f32>(2.2, 2.2, 2.2));
    return out;
}



// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(in.colour, 1.0);
}