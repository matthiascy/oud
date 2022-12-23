struct CameraUniform {
    view_proj: mat4x4<f32>
}

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
}

struct InstanceInput {
    @location(5) model_mat_c0: vec4<f32>,
    @location(6) model_mat_c1: vec4<f32>,
    @location(7) model_mat_c2: vec4<f32>,
    @location(8) model_mat_c3: vec4<f32>,
}

struct VertexOutput {
    // position in WebGPU normalized device coordinate space.
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) tex_coord: vec2<f32>,
}

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var model_mat = mat4x4<f32>(
        instance.model_mat_c0,
        instance.model_mat_c1,
        instance.model_mat_c2,
        instance.model_mat_c3
    );

    var out: VertexOutput;
    out.color = vertex.color;
    out.tex_coord = vertex.tex_coord;
    out.clip_position = camera.view_proj * model_mat * vec4<f32>(vertex.position, 1.0);
    return out;
}

@group(0) @binding(0)
var tex: texture_2d<f32>;
@group(0) @binding(1)
var spl: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // @location(0): stores the output in the first output location(color attachment 0)
    // in fragment shader, @bulitin(position) is the framebuffer coordinate.
    //return vec4<f32>(in.color, 1.0);
    return textureSample(tex, spl, in.tex_coord);
}
