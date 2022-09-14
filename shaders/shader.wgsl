// Vertex shader
struct VertexInput {
    @location(0) abs_position: vec3<i32>,
    @location(1) rel_position: vec3<f32>,
    @location(2) colour: vec4<f32>,
    @location(3) normal: vec3<f32>
};


struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) colour: vec4<f32>,
};


struct CameraUniform {
    view_proj: mat4x4<f32>,
    player_position: vec3<i32>,
    sun_normal: vec3<f32>,
    sun_intensity: f32,
}

struct ChunkIDUniform {
    chunk_id: vec3<i32>,
}

@group(0) @binding(0) // 1.
var<uniform> camera: CameraUniform;


@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let camera_position = vec3<f32>(in.abs_position - camera.player_position) + in.rel_position;
    
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4(camera_position, 1.0);
    let dot_product = dot(in.normal, camera.sun_normal);
    let light_level = clamp((dot_product+1.0)/2.0, 0.1, 1.0) ;
    let rgba = pow(in.colour.xyz * light_level, vec3<f32>(2.2, 2.2, 2.2));
    out.colour = vec4(rgba, in.colour.w);
    return out;
}



// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.colour);
}