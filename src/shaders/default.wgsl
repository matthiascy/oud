struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
}

struct VertexOutput {
    // position in WebGPU normalized device coordinate space.
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) tex_coord: vec2<f32>,
}

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.color = vertex.color;
    out.tex_coord = vertex.tex_coord;
    out.clip_position = vec4<f32>(vertex.position, 1.0);
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
    // vec4<f32>(in.color, 1.0);
    return textureSample(tex, spl, in.tex_coord);
}
