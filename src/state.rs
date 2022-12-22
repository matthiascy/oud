use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;

use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::window::Window;

use crate::camera::{Camera, CameraController, CameraUniform};
use crate::texture::Texture;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub tex_coords: [f32; 2],
}

impl Vertex {
    pub const SIZE: wgpu::BufferAddress = std::mem::size_of::<Vertex>() as wgpu::BufferAddress;

    pub const fn layout<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: Self::SIZE,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ], // wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        }
    }
}

type Idx = u32;
const IDX_SIZE: wgpu::BufferAddress = std::mem::size_of::<Idx>() as wgpu::BufferAddress;
const IDX_FORMAT: wgpu::IndexFormat = wgpu::IndexFormat::Uint32;

const TRIANGLE_VERTICES: &[Vertex] = &[
    Vertex {
        position: [-0.5, -0.5, 0.0],
        color: [1.0, 0.0, 0.0],
        tex_coords: [0.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.0],
        color: [0.0, 1.0, 0.0],
        tex_coords: [1.0, 0.0],
    },
    Vertex {
        position: [0.0, 0.5, 0.0],
        color: [0.0, 0.0, 1.0],
        tex_coords: [0.5, 1.0],
    },
];

const TRIANGLE_INDICES: &[Idx] = &[0, 1, 2];

const HEXAGON_VERTICES: &[Vertex] = &[
    Vertex {
        position: [0.5, 0.5, 0.0],
        color: [0.0, 0.0, 1.0],
        tex_coords: [0.3, 0.4],
    },
    Vertex {
        position: [0.8, 0.3, 0.0],
        color: [0.0, 1.0, 0.0],
        tex_coords: [0.6, 0.2],
    },
    Vertex {
        position: [0.5, 0.2, 0.0],
        color: [0.0, 0.0, 1.0],
        tex_coords: [0.3, 0.0],
    },
    Vertex {
        position: [0.2, 0.3, 0.0],
        color: [1.0, 0.0, 0.0],
        tex_coords: [0.0, 0.2],
    },
    Vertex {
        position: [0.2, 0.7, 0.0],
        color: [1.0, 0.0, 0.0],
        tex_coords: [0.0, 0.6],
    },
    Vertex {
        position: [0.5, 0.8, 0.0],
        color: [0.0, 0.0, 1.0],
        tex_coords: [0.3, 1.0],
    },
    Vertex {
        position: [0.8, 0.7, 0.0],
        color: [0.0, 1.0, 0.0],
        tex_coords: [0.6, 0.8],
    },
];

const HEXAGON_INDICES: &[Idx] = &[0, 1, 6, 0, 2, 1, 0, 3, 2, 0, 4, 3, 0, 5, 4, 0, 6, 5];

pub struct SlicedBuffer {
    buf: wgpu::Buffer,
    #[allow(dead_code)]
    cap: wgpu::BufferAddress,
    slices: Vec<Range<wgpu::BufferAddress>>,
}

const INITIAL_VERTEX_BUFFER_SIZE: wgpu::BufferAddress = Vertex::SIZE * 1024;
const INITIAL_INDEX_BUFFER_SIZE: wgpu::BufferAddress = IDX_SIZE * 1024;

pub struct RenderState {
    #[allow(dead_code)]
    instance: wgpu::Instance,
    #[allow(dead_code)]
    adapter: wgpu::Adapter,
    pub surface: wgpu::Surface,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub size: PhysicalSize<u32>,
    clear_color: wgpu::Color,
    pipelines: HashMap<&'static str, wgpu::RenderPipeline>,
    pub current_pipeline: &'static str,
    pub vertex_buffer: SlicedBuffer,
    pub index_buffer: SlicedBuffer,
    textures: Vec<Texture>,
    bind_groups: HashMap<&'static str, Vec<wgpu::BindGroup>>,
    texture_idx: usize,
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_uniform_buffer: wgpu::Buffer,
}

impl RenderState {
    pub async fn new(window: &Window, clear_color: wgpu::Color) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };

        // let adapter = instance
        //     .enumerate_adapters(wgpu::Backends::all())
        //     .filter(|adapter| {
        //         println!("{:?}", adapter.get_info());
        //         !surface.get_supported_formats(&adapter).is_empty()
        //     })
        //     .next()
        //     .unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device"),
                    features: wgpu::Features::empty(),
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                },
                None,
            )
            .await
            .unwrap();

        // adapter.features(); device.features();

        let surface_config = {
            let formats = surface.get_supported_formats(&adapter);
            println!("Supported surface formats:");
            for format in &formats {
                println!("  - {:?}", format);
            }

            let present_modes = surface.get_supported_present_modes(&adapter);
            println!("Supported surface present modes:");
            for mode in &present_modes {
                println!("  - {:?}", mode);
            }

            wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: formats[0],
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
            }
        };
        surface.configure(&device, &surface_config);

        let mut vertex_buffer = {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vertex-buffer"),
                size: INITIAL_VERTEX_BUFFER_SIZE,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            SlicedBuffer {
                buf,
                cap: INITIAL_VERTEX_BUFFER_SIZE,
                slices: Vec::new(),
            }
        };

        let mut offset: wgpu::BufferAddress = 0;
        let triangle_vertex_buffer_size =
            TRIANGLE_VERTICES.len() as wgpu::BufferAddress * Vertex::SIZE;
        vertex_buffer
            .slices
            .push(offset..offset + triangle_vertex_buffer_size);
        queue.write_buffer(
            &vertex_buffer.buf,
            0,
            bytemuck::cast_slice(TRIANGLE_VERTICES),
        );
        offset += triangle_vertex_buffer_size;
        let hexagon_vertex_buffer_size =
            HEXAGON_VERTICES.len() as wgpu::BufferAddress * Vertex::SIZE;
        vertex_buffer
            .slices
            .push(offset..offset + hexagon_vertex_buffer_size);
        queue.write_buffer(
            &vertex_buffer.buf,
            offset,
            bytemuck::cast_slice(HEXAGON_VERTICES),
        );

        println!("preparing index buffer");
        let mut index_buffer = {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("index-buffer"),
                size: INITIAL_INDEX_BUFFER_SIZE,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            SlicedBuffer {
                buf,
                cap: INITIAL_INDEX_BUFFER_SIZE,
                slices: Vec::new(),
            }
        };
        offset = 0;
        index_buffer.slices.push(offset..offset + 3 * IDX_SIZE);
        queue.write_buffer(
            &index_buffer.buf,
            offset,
            bytemuck::cast_slice(&TRIANGLE_INDICES),
        );
        offset += 3 * IDX_SIZE;
        index_buffer
            .slices
            .push(offset..offset + HEXAGON_INDICES.len() as wgpu::BufferAddress * IDX_SIZE);
        queue.write_buffer(
            &index_buffer.buf,
            offset,
            bytemuck::cast_slice(&HEXAGON_INDICES),
        );

        let textures = {
            #[cfg(target_arch = "wasm32")]
            {
                log::info!("loading textures");
                vec![
                    Texture::from_bytes(
                        &device,
                        &queue,
                        include_bytes!("../assets/cliff.jpg"),
                        "texture-cliff",
                    ),
                    Texture::from_bytes(
                        &device,
                        &queue,
                        include_bytes!("../assets/brick.jpg"),
                        "texture-brick",
                    ),
                ]
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                vec![
                    Texture::from_file(
                        &device,
                        &queue,
                        Path::new("./assets/cliff.jpg"),
                        "tex-cliff",
                    ),
                    Texture::from_file(
                        &device,
                        &queue,
                        Path::new("./assets/ground.jpg"),
                        "tex-ground",
                    ),
                ]
            }
        };

        let camera = Camera {
            eye: glam::Vec3::new(0.0, 1.0, 2.0),
            target: glam::Vec3::ZERO,
            up: glam::Vec3::Y,
            aspect: surface_config.width as f32 / surface_config.height as f32,
            vfov: 45.0_f32,
            near: 0.1,
            far: 100.0,
        };

        let camera_controller = CameraController::new(0.1);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        use wgpu::util::DeviceExt;

        let camera_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera-uniform-buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("texture-bind-group-layout"),
                entries: &[
                    // Sampled texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Texture sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // Matches the filterable field of the texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let camera_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera-uniform-bind-group-layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let texture_bind_groups = vec![
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bind-group-texture-0"),
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&textures[0].view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&textures[0].sampler),
                    },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bind-group-texture-1"),
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&textures[1].view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&textures[1].sampler),
                    },
                ],
            }),
        ];

        let camera_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera-uniform-bind-group"),
            layout: &camera_uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_uniform_buffer.as_entire_binding(),
            }],
        });

        let default_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("default-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/default.wgsl").into()),
        });

        let variant_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("variant-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/variant.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("default-render-pipeline-layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_uniform_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let default_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("default-render-pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &default_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::layout()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false, // requires Features::DEPTH_CLIP_CONTROL
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false, // requires Features::CONSERVATIVE_RASTERIZATION
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &default_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::Zero,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::Zero,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None, // render to array textures
        });

        let variant_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("variant-render-pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &variant_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::layout()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false, // requires Features::DEPTH_CLIP_CONTROL
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false, // requires Features::CONSERVATIVE_RASTERIZATION
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &variant_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None, // render to array textures
        });

        let mut pipelines = HashMap::new();
        pipelines.insert("default", default_pipeline);
        pipelines.insert("variant", variant_pipeline);

        let mut bind_groups = HashMap::new();
        bind_groups.insert("texture", texture_bind_groups);
        bind_groups.insert("camera", vec![camera_uniform_bind_group]);

        Self {
            surface,
            device,
            queue,
            size,
            surface_config,
            instance,
            adapter,
            clear_color,
            pipelines,
            current_pipeline: "default",
            vertex_buffer,
            index_buffer,
            textures,
            bind_groups,
            texture_idx: 0,
            camera,
            camera_controller,
            camera_uniform,
            camera_uniform_buffer,
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    #[allow(unused_variables)]
    pub fn handle_input(&mut self, event: &WindowEvent) -> Response {
        if self.camera_controller.process_events(event).is_ignored() {
            match event {
                WindowEvent::MouseInput { state, button, .. } => {
                    print!("MouseInput: {:?} {:?}\r", state, button);
                    Response::Handled
                }
                WindowEvent::CursorMoved { position, .. } => {
                    print!("CursorMoved: {:?}\r", position);
                    self.clear_color = wgpu::Color {
                        r: position.x as f64 / self.size.width as f64,
                        g: position.y as f64 / self.size.height as f64,
                        b: 0.5,
                        a: 1.0,
                    };
                    Response::Handled
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    print!("KeyboardInput: {:?}\r", input);
                    match input {
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(key),
                            ..
                        } => match key {
                            VirtualKeyCode::Space => {
                                self.current_pipeline = match self.current_pipeline {
                                    "default" => "variant",
                                    "variant" => "default",
                                    _ => "default",
                                };
                                Response::Handled
                            }
                            VirtualKeyCode::T => {
                                self.texture_idx = (self.texture_idx + 1) % self.textures.len();
                                Response::Handled
                            }
                            _ => Response::Ignored,
                        },
                        _ => Response::Ignored,
                    }
                }
                _ => Response::Ignored,
            }
        } else {
            Response::Ignored
        }
    }

    pub fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("main-render-encoder"),
            });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main-render-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.pipelines[self.current_pipeline]);
            render_pass.set_bind_group(0, &self.bind_groups["texture"][self.texture_idx], &[]);
            render_pass.set_bind_group(1, &self.bind_groups["camera"][0], &[]);
            render_pass.set_index_buffer(self.index_buffer.buf.slice(..), IDX_FORMAT);

            #[cfg(not(target_arch = "wasm32"))]
            {
                // Only bind once the vertex buffer and index buffer,
                // then draw the different slices with vertex and index offset.
                render_pass.set_vertex_buffer(0, self.vertex_buffer.buf.slice(..));
                self.index_buffer
                    .slices
                    .iter()
                    .enumerate()
                    .for_each(|(i, index_range)| {
                        let count = ((index_range.end - index_range.start) / IDX_SIZE) as u32;
                        let index_offset = (index_range.start / IDX_SIZE) as u32;
                        let vertex_offset = self.vertex_buffer.slices[i].start / Vertex::SIZE;
                        render_pass.draw_indexed(
                            index_offset..index_offset + count,
                            vertex_offset as i32,
                            0..1,
                        );
                    });
            }

            #[cfg(target_arch = "wasm32")]
            {
                // Draw elements with base vertex is not supported using webgl.
                self.index_buffer
                    .slices
                    .iter()
                    .zip(self.vertex_buffer.slices.iter())
                    .for_each(|(index_range, vertex_range)| {
                        let vertex_buffer = self.vertex_buffer.buf.slice(vertex_range.clone());
                        let index_offset = (index_range.start / IDX_SIZE) as u32;
                        let count = (index_range.end - index_range.start) / IDX_SIZE;
                        render_pass.set_vertex_buffer(0, vertex_buffer);
                        render_pass.draw_indexed(
                            index_offset..index_offset + count as u32,
                            0,
                            0..1,
                        );
                    });
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        Ok(())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Response {
    Handled,
    Ignored,
}

impl Response {
    pub fn is_handled(self) -> bool {
        self == Self::Handled
    }

    pub fn is_ignored(self) -> bool {
        self == Self::Ignored
    }
}
