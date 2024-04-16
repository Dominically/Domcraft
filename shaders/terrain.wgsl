// Vertex shader
struct VertexInput {
    @location(0) abs_position: vec3<i32>,
    @location(1) rel_position: vec3<f32>,
    @location(2) colour: vec4<f32>,
    @location(3) normal: vec3<f32>
};


struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) colour: vec4<f32>, //Colour from ambient and specular shading.
    @location(1) cam_reflect: vec3<f32>,
};


struct CameraUniform {
    view_proj: mat4x4<f32>,
    position_abs: vec3<i32>,
    position_rel: vec3<f32>,
    sun_normal: vec3<f32>,
    sun_intensity: f32,
}

struct CameraFragmentUniform {
    sun_normal: vec3<f32>,
    sun_intensity: f32
}

@group(0) @binding(0) // 1.
var<uniform> camera: CameraUniform;

@group(1) @binding(0) //don't know if this will work.
var<uniform> camera_fragment: CameraFragmentUniform;


@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let camera_relative = vec3<f32>(in.abs_position - camera.position_abs) + in.rel_position - camera.position_rel;
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4(camera_relative, 1.0);
    let dot_product = dot(in.normal, camera.sun_normal);
    // let diffuse_level = clamp((dot_product+1.0)/2.0, 0.1, 1.0) * camera.sun_intensity;
    let diffuse_level = clamp(clamp(dot_product/2.0 + 0.5, 0.0, 1.0) * camera.sun_intensity + 0.2, 0.0, 1.0);

    //https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector
    let reflection = camera_relative - (2.0 * dot(camera_relative, in.normal) * in.normal);

    // let rgba = pow(in.colour.xyz * diffuse_level, vec3<f32>(2.2, 2.2, 2.2));
    let rgba = in.colour.xyz * diffuse_level;
    out.colour = vec4(rgba, in.colour.w);

    // let colour_unclamped = vec4<f32>(reflection/128.0, 1.0);
    // out.colour = clamp(colour_unclamped, vec4<f32>(0.0), vec4<f32>(1.0));

    out.cam_reflect = reflection;
    return out;
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // var light_intensity = 1.0 - pow(2.7183, dot(normalize(in.cam_reflect), normalize(vec3<f32>(0.0, -0.64, -0.77))));
    // light_intensity = clamp(light_intensity, 0.0, 1.0); 
    // let in_rgb = in.colour.xyz;
    // let diff = vec3<f32>(light_intensity, light_intensity, light_intensity); //this can probably be done better but the wgsl spec is hard to read
    // let colour = in.colour + (vec4(1.0, 1.0, 1.0, 1.0) - in.colour) * light_intensity;

    // return vec4<f32>(in.colour);
     
    //Specular intensity.
    // let sun_dir = vec3<f32>(0.5773502691896257, 0.5773502691896257, 0.5773502691896257);
    let reflect_norm = normalize(in.cam_reflect);
    let spec_intensity = dot(reflect_norm, camera_fragment.sun_normal);
    let spec = pow(spec_intensity / 2.0 + 0.5, 8.0)/4.0 * camera_fragment.sun_intensity;
    let specular_add = vec4<f32>(spec, spec, spec, 0.0);
    
    let pre_gamma_colour =  clamp(in.colour + specular_add, vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(1.0, 1.0, 1.0, 1.0));
    let colour = pow(pre_gamma_colour, vec4<f32>(2.2, 2.2, 2.2, 1.0));

    return colour;
}