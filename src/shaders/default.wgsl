struct VertexOutput {
    // position in WebGPU normalized device coordinate space.
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) in_vert_idx: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(1 - i32(in_vert_idx)) * 0.5;
    let y = f32(i32(in_vert_idx & 1u) * 2 - 1) * 0.5;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // @location(0): stores the output in the first output location(color attachment 0)
    // in fragment shader, @bulitin(position) is the framebuffer coordinate.
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
