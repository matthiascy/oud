use std::collections::HashMap;
use std::ops::Range;

use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::window::Window;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
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
            ], // wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        }
    }
}

const TRIANGLE_VERTICES: &[Vertex] = &[
    Vertex {
        position: [-0.5, -0.5, 0.0],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.0],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        position: [0.0, 0.5, 0.0],
        color: [0.0, 0.0, 1.0],
    },
];

const HEXAGON_VERTICES: &[Vertex] = &[
    Vertex {
        position: [1.0, 0.0, 0.0],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.0],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        position: [-0.5, -0.5, 0.0],
        color: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [-1.0, 0.0, 0.0],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        position: [-0.5, 0.5, 0.0],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        position: [0.5, 0.5, 0.0],
        color: [0.0, 0.0, 1.0],
    },
];

pub struct SlicedBuffer {
    buf: wgpu::Buffer,
    #[allow(dead_code)]
    cap: wgpu::BufferAddress,
    slices: Vec<Range<wgpu::BufferAddress>>,
}

const INITIAL_VERTEX_BUFFER_SIZE: wgpu::BufferAddress =
    std::mem::size_of::<Vertex>() as wgpu::BufferAddress * 1024;
const INITIAL_INDEX_BUFFER_SIZE: wgpu::BufferAddress =
    std::mem::size_of::<u32>() as wgpu::BufferAddress * 1024;

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
        offset += triangle_vertex_buffer_size;

        let hexagon_vertex_buffer_size =
            HEXAGON_VERTICES.len() as wgpu::BufferAddress * Vertex::SIZE;
        vertex_buffer
            .slices
            .push(offset..offset + hexagon_vertex_buffer_size);

        queue.write_buffer(
            &vertex_buffer.buf,
            0,
            bytemuck::cast_slice(TRIANGLE_VERTICES),
        );

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

        index_buffer
            .slices
            .push(0..3 * std::mem::size_of::<u32>() as wgpu::BufferAddress);

        queue.write_buffer(&index_buffer.buf, 0, bytemuck::cast_slice(&[0, 1, 2]));

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
                bind_group_layouts: &[],
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
                        virtual_keycode: Some(VirtualKeyCode::Space),
                        ..
                    } => {
                        self.current_pipeline = match self.current_pipeline {
                            "default" => "variant",
                            "variant" => "default",
                            _ => "default",
                        };
                        Response::Handled
                    }
                    _ => Response::Ignored,
                }
            }
            _ => Response::Ignored,
        }
    }

    pub fn update(&mut self) {}

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
            for vertex_range in &self.vertex_buffer.slices {
                let buffer = self.vertex_buffer.buf.slice(vertex_range.clone());
                let count = (vertex_range.end - vertex_range.start) / Vertex::SIZE;
                render_pass.set_vertex_buffer(0, buffer);
                //   render_pass.draw(0..count as u32, 0..1);
                render_pass.draw_indexed(indices, base_vertex, instances)
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
