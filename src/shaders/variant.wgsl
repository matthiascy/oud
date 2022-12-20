struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) position: vec2<f32>,
    @location(1) color: vec3<f32>,
}

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(vertex.position, 1.0);
    out.position = vec2<f32>(vertex.position.x, vertex.position.y);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // @location(0): stores the output in the first output location(color attachment 0)
    // in fragment shader, @bulitin(position) is the framebuffer coordinate.
    return vec4<f32>(in.position, 0.5, 1.0);
}
