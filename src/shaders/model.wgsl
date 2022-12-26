struct CameraUniform {
    view_proj: mat4x4<f32>,
}

struct VInput {
    @location(0) position: vec3<f32>,
    @location(1) texcoord: vec2<f32>,
    @location(2) normal: vec3<f32>,
}

struct VOutput {
    @builtin(position) ndc_position: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
    @location(1) normal: vec3<f32>,
}

struct InstanceInput {
    @location(3) model_mat_col_0: vec4<f32>,
    @location(4) model_mat_col_1: vec4<f32>,
    @location(5) model_mat_col_2: vec4<f32>,
    @location(6) model_mat_col_3: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@vertex
fn vs_main(v: VInput, instance: InstanceInput) -> VOutput {
    let model_mat = mat4x4<f32>(
        instance.model_mat_col_0,
        instance.model_mat_col_1,
        instance.model_mat_col_2,
        instance.model_mat_col_3
    );
    var output: VOutput;
    output.ndc_position = camera.view_proj * model_mat * vec4<f32>(v.position, 1.0);
    output.texcoord = v.texcoord;
    output.normal = v.normal;
    return output;
}

@group(1) @binding(0)
var tex: texture_2d<f32>;
@group(1) @binding(1)
var spl: sampler;

@fragment
fn fs_main(v: VOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(tex, spl, v.texcoord);
    return tex_color;
}
