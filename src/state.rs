use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::EventLoopWindowTarget;
use winit::window::Window;

use crate::assets;
use crate::buffer::SlicedBuffer;
use crate::camera::{Camera, CameraController, CameraUniform};
use crate::gui::{GuiRenderer, ScreenDescriptor};
use crate::model::{DrawModel, Model, Vertex, VertexPCT, VertexPTN};
use crate::texture::Texture;

type Idx = u32;
const IDX_SIZE: wgpu::BufferAddress = std::mem::size_of::<Idx>() as wgpu::BufferAddress;
const IDX_FORMAT: wgpu::IndexFormat = wgpu::IndexFormat::Uint32;

const TRIANGLE_VERTICES: &[VertexPCT] = &[
    VertexPCT {
        position: [-0.5, -0.5, 0.0],
        color: [1.0, 0.0, 0.0],
        texcoord: [0.0, 0.0],
    },
    VertexPCT {
        position: [0.5, -0.5, 0.0],
        color: [0.0, 1.0, 0.0],
        texcoord: [1.0, 0.0],
    },
    VertexPCT {
        position: [0.0, 0.5, 0.0],
        color: [0.0, 0.0, 1.0],
        texcoord: [0.5, 1.0],
    },
];

const TRIANGLE_INDICES: &[Idx] = &[0, 1, 2];

const HEXAGON_VERTICES: &[VertexPCT] = &[
    VertexPCT {
        position: [0.5, 0.5, 0.0],
        color: [0.0, 0.0, 1.0],
        texcoord: [0.3, 0.4],
    },
    VertexPCT {
        position: [0.8, 0.3, 0.0],
        color: [0.0, 1.0, 0.0],
        texcoord: [0.6, 0.2],
    },
    VertexPCT {
        position: [0.5, 0.2, 0.0],
        color: [0.0, 0.0, 1.0],
        texcoord: [0.3, 0.0],
    },
    VertexPCT {
        position: [0.2, 0.3, 0.0],
        color: [1.0, 0.0, 0.0],
        texcoord: [0.0, 0.2],
    },
    VertexPCT {
        position: [0.2, 0.7, 0.0],
        color: [1.0, 0.0, 0.0],
        texcoord: [0.0, 0.6],
    },
    VertexPCT {
        position: [0.5, 0.8, 0.0],
        color: [0.0, 0.0, 1.0],
        texcoord: [0.3, 1.0],
    },
    VertexPCT {
        position: [0.8, 0.7, 0.0],
        color: [0.0, 1.0, 0.0],
        texcoord: [0.6, 0.8],
    },
];

const HEXAGON_INDICES: &[Idx] = &[0, 1, 6, 0, 2, 1, 0, 3, 2, 0, 4, 3, 0, 5, 4, 0, 6, 5];

const INITIAL_VERTEX_BUFFER_SIZE: wgpu::BufferAddress = VertexPCT::SIZE * 1024;
const INITIAL_INDEX_BUFFER_SIZE: wgpu::BufferAddress = IDX_SIZE * 1024;

#[derive(Debug, Clone, Copy)]
pub struct InstanceParams {
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
}

impl InstanceParams {
    const GPU_DATA_SIZE: wgpu::BufferAddress =
        std::mem::size_of::<[f32; 16]>() as wgpu::BufferAddress;

    pub fn new(position: glam::Vec3, rotation: glam::Quat, scale: glam::Vec3) -> Self {
        Self {
            position,
            rotation,
            scale,
        }
    }

    pub fn into_array(self) -> [f32; 16] {
        glam::Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
            .to_cols_array()
        //        glam::Mat4::from_rotation_translation(self.rotation, self.position).to_cols_array()
    }

    pub const fn layout<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: Self::GPU_DATA_SIZE,
            step_mode: wgpu::VertexStepMode::Instance, // step mode is per instance
            attributes: &[
                // 4 coloumns of the matrix
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5, // not conflicting with the Vertex attributes buffer
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }

    pub const fn model_shader_layout<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: Self::GPU_DATA_SIZE,
            step_mode: wgpu::VertexStepMode::Instance, // step mode is per instance
            attributes: &[
                // 4 coloumns of the matrix
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3, // not conflicting with the Vertex attributes buffer
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

const NUM_INSTANCES_PER_ROW: usize = 9;

/// Action to be performed as consequence of a [`wgpu::SurfaceError`]
pub enum SurfaceErrorAction {
    /// Do nothing and skip the current frame
    Skip,
    /// Recreate the surface, then skip the current frame.
    Recreate,
}

pub struct GpuConfiguration {
    /// Device requirements for requesting a device.
    pub device_descriptor: wgpu::DeviceDescriptor<'static>,
    /// Backend API to use.
    pub backends: wgpu::Backends,
    /// Present mode used for the primary swapchain(surface).
    pub present_mode: wgpu::PresentMode,
    /// Power preference for the GPU.
    pub power_preference: wgpu::PowerPreference,
    /// Texture format for the depth buffer.
    pub depth_format: Option<wgpu::TextureFormat>,
    /// Callback for handling surface errors.
    pub on_surface_error: Arc<dyn Fn(wgpu::SurfaceError) -> SurfaceErrorAction>,
}

impl Default for GpuConfiguration {
    fn default() -> Self {
        Self {
            device_descriptor: wgpu::DeviceDescriptor {
                label: Some("wgpu_device"),
                features: wgpu::Features::default(),
                limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
            },
            backends: wgpu::Backends::PRIMARY | wgpu::Backends::GL,
            present_mode: wgpu::PresentMode::AutoVsync,
            power_preference: wgpu::PowerPreference::HighPerformance,
            depth_format: None,
            on_surface_error: Arc::new(|err| {
                if err == wgpu::SurfaceError::Outdated {
                    // This error occurs when the app is minimized on Windows.
                    // Silently return here to prevent spamming the console with:
                    // "The underlying surface has changed, and therefore the swapchain must be updated."
                } else {
                    log::warn!("Dropped frame with error: {err}");
                }
                SurfaceErrorAction::Skip
            }),
        }
    }
}

/// Chooses a preferred texture format for the swapchain.
///
/// Prefers linear color space if availale, otherwise fall back to the first available format.
pub fn preferred_surface_format(formats: &[wgpu::TextureFormat]) -> wgpu::TextureFormat {
    for &format in formats {
        match format {
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm => {
                return format;
            }
            _ => (),
        }
    }
    formats[0]
}

pub struct SurfaceState {
    surface: wgpu::Surface,
    config: wgpu::SurfaceConfiguration,
}

impl SurfaceState {
    /// Physical width of the surface in pixels.
    pub fn width(&self) -> u32 {
        self.config.width
    }

    /// Physical height of the surface in pixels.
    pub fn height(&self) -> u32 {
        self.config.height
    }

    pub fn config(&self) -> &wgpu::SurfaceConfiguration {
        &self.config
    }

    pub fn surface(&self) -> &wgpu::Surface {
        &self.surface
    }

    /// Resizes the surface and updates the swapchain configuration.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.config.width != width || self.config.height != height {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(device, &self.config);
        }
    }

    /// Re-configures the surface with the known configuration.
    pub fn reconfigure(&self, device: &wgpu::Device) {
        self.surface.configure(device, &self.config);
    }
}

pub struct RenderState {
    #[allow(dead_code)]
    instance: wgpu::Instance,
    #[allow(dead_code)]
    adapter: wgpu::Adapter,

    gpu_config: GpuConfiguration,

    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_state: SurfaceState,

    clear_color: wgpu::Color,
    pipelines: HashMap<&'static str, wgpu::RenderPipeline>,
    pub current_pipeline: &'static str,
    pub vertex_buffer: SlicedBuffer,
    pub index_buffer: SlicedBuffer,
    textures: HashMap<&'static str, Vec<Texture>>,
    bind_groups: HashMap<&'static str, Vec<wgpu::BindGroup>>,
    current_texture_idx: usize,
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_uniform_buffer: wgpu::Buffer,
    object_instances: Vec<InstanceParams>,
    object_instances_buffer: wgpu::Buffer,
    model: Model,

    gui_ctx: egui::Context,
    gui_state: egui_winit::State,
    gui_renderer: GuiRenderer,
}

impl RenderState {
    pub async fn new(
        window: &Window,
        event_loop: &EventLoopWindowTarget<()>,
        clear_color: wgpu::Color,
    ) -> Self {
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

        let gpu_config = GpuConfiguration::default();

        let (device, queue) = adapter
            .request_device(&gpu_config.device_descriptor, None)
            .await
            .unwrap();

        // adapter.features(); device.features();

        let surface_config = {
            let size = window.inner_size();
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

        let mut vertex_buffer = SlicedBuffer::new(
            &device,
            INITIAL_VERTEX_BUFFER_SIZE,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            Some("vertex-buffer"),
        );

        let mut offset: wgpu::BufferAddress = 0;
        let triangle_vertex_buffer_size =
            TRIANGLE_VERTICES.len() as wgpu::BufferAddress * VertexPCT::SIZE;
        {
            let ref mut this = vertex_buffer;
            let range = offset..offset + triangle_vertex_buffer_size;
            assert!(range.start < range.end && range.end <= this.capacity());
            this.subslices_mut().push(range);
        };
        queue.write_buffer(
            &vertex_buffer.buffer(),
            0,
            bytemuck::cast_slice(TRIANGLE_VERTICES),
        );
        offset += triangle_vertex_buffer_size;
        let hexagon_vertex_buffer_size =
            HEXAGON_VERTICES.len() as wgpu::BufferAddress * VertexPCT::SIZE;
        {
            let ref mut this = vertex_buffer;
            let range = offset..offset + hexagon_vertex_buffer_size;
            assert!(range.start < range.end && range.end <= this.capacity());
            this.subslices_mut().push(range);
        };
        queue.write_buffer(
            &vertex_buffer.buffer(),
            offset,
            bytemuck::cast_slice(HEXAGON_VERTICES),
        );

        let mut index_buffer = SlicedBuffer::new(
            &device,
            INITIAL_INDEX_BUFFER_SIZE,
            wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            Some("index-buffer"),
        );
        offset = 0;
        index_buffer
            .subslices_mut()
            .push(offset..offset + 3 * IDX_SIZE);
        queue.write_buffer(
            &index_buffer.buffer(),
            offset,
            bytemuck::cast_slice(&TRIANGLE_INDICES),
        );
        offset += 3 * IDX_SIZE;
        index_buffer
            .subslices_mut()
            .push(offset..offset + HEXAGON_INDICES.len() as wgpu::BufferAddress * IDX_SIZE);
        queue.write_buffer(
            &index_buffer.buffer(),
            offset,
            bytemuck::cast_slice(&HEXAGON_INDICES),
        );

        const DISPLACEMENT: glam::Vec3 = glam::Vec3::new(
            NUM_INSTANCES_PER_ROW as f32,
            0.0,
            NUM_INSTANCES_PER_ROW as f32,
        );

        let object_instances = (0..NUM_INSTANCES_PER_ROW)
            .flat_map(|z| {
                (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                    let n = (x + z * NUM_INSTANCES_PER_ROW) as f32;
                    let position =
                        glam::Vec3::new(x as f32 * 2.0, -0.5, z as f32 * 2.0) - DISPLACEMENT;
                    let rotation = glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_8 * n);
                    InstanceParams::new(position, rotation, glam::Vec3::new(0.5, 0.5, 0.5))
                })
            })
            .collect::<Vec<_>>();
        let object_instances_data = object_instances
            .iter()
            .map(|params| params.into_array())
            .collect::<Vec<_>>();
        let object_instances_buffer: wgpu::Buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("obj-instances-buffer"),
                contents: bytemuck::cast_slice(&object_instances_data),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let mut textures = HashMap::new();
        textures.insert("binding", {
            #[cfg(target_arch = "wasm32")]
            {
                log::info!("loading textures");
                vec![
                    Texture::from_bytes(
                        &device,
                        &queue,
                        include_bytes!("../assets/cliff.jpg"),
                        "texture-cliff",
                    )
                    .unwrap(),
                    Texture::from_bytes(
                        &device,
                        &queue,
                        include_bytes!("../assets/brick.jpg"),
                        "texture-brick",
                    )
                    .unwrap(),
                    Texture::from_bytes(
                        &device,
                        &queue,
                        include_bytes!("../assets/cube-diffuse.jpg"),
                        "texture-cube",
                    )
                    .unwrap(),
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
                    )
                    .unwrap(),
                    Texture::from_file(
                        &device,
                        &queue,
                        Path::new("./assets/ground.jpg"),
                        "tex-ground",
                    )
                    .unwrap(),
                    Texture::from_file(
                        &device,
                        &queue,
                        Path::new("./assets/damascus.jpg"),
                        "tex-damascus",
                    )
                    .unwrap(),
                ]
            }
        });
        textures.insert(
            "depth",
            vec![Texture::create_depth_texture(
                &device,
                surface_config.width,
                surface_config.height,
                "depth-texture",
            )],
        );

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

        let depth_map_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("depth-map-bind-group-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                ],
            });

        let depth_map_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("depth-map-bind-group"),
            layout: &depth_map_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&textures["depth"][0].view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&textures["depth"][0].sampler),
                },
            ],
        });

        let texture_bind_groups = vec![
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bind-group-texture-0"),
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&textures["binding"][0].view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&textures["binding"][0].sampler),
                    },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bind-group-texture-1"),
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&textures["binding"][1].view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&textures["binding"][1].sampler),
                    },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bind-group-texture-1"),
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&textures["binding"][2].view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&textures["binding"][2].sampler),
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

        let model = assets::load_model("cube.obj", &device, &queue, &texture_bind_group_layout)
            .await
            .unwrap();

        let default_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("default-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/default.wgsl").into()),
        });

        let variant_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("variant-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/variant.wgsl").into()),
        });

        let model_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("model-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/model.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("default-render-pipeline-layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_uniform_bind_group_layout,
                    &depth_map_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let default_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("default-render-pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &default_shader,
                entry_point: "vs_main",
                buffers: &[InstanceParams::layout(), VertexPCT::buffer_layout()],
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
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
                buffers: &[VertexPCT::buffer_layout()],
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

        let model_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("model-render-pipeline-layout"),
                bind_group_layouts: &[
                    &camera_uniform_bind_group_layout,
                    &texture_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let model_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("model-render-pipeline"),
            layout: Some(&model_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &model_shader,
                entry_point: "vs_main",
                buffers: &[
                    VertexPTN::buffer_layout(),
                    InstanceParams::model_shader_layout(),
                ],
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &model_shader,
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
            multiview: None,
        });

        let mut pipelines = HashMap::new();
        pipelines.insert("default", default_pipeline);
        pipelines.insert("variant", variant_pipeline);
        pipelines.insert("model", model_pipeline);

        let mut bind_groups = HashMap::new();
        bind_groups.insert("texture", texture_bind_groups);
        bind_groups.insert("camera", vec![camera_uniform_bind_group]);
        bind_groups.insert("depth_map", vec![depth_map_bind_group]);

        let gui_ctx = egui::Context::default();
        let gui_state = egui_winit::State::new(event_loop);
        let gui_renderer = GuiRenderer::new(&device, surface_config.format, None, 1);

        Self {
            device,
            queue,
            surface_state: SurfaceState {
                surface,
                config: surface_config,
            },
            instance,
            adapter,
            clear_color,
            pipelines,
            current_pipeline: "default",
            vertex_buffer,
            index_buffer,
            textures,
            bind_groups,
            current_texture_idx: 0,
            camera,
            camera_controller,
            camera_uniform,
            camera_uniform_buffer,
            object_instances,
            object_instances_buffer,
            model,
            gpu_config,
            gui_ctx,
            gui_state,
            gui_renderer,
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_state
                .resize(&self.device, new_size.width, new_size.height);
        }
        if let Some(depth_textures) = self.textures.get_mut("depth") {
            depth_textures[0] = Texture::create_depth_texture(
                &self.device,
                self.surface_state.config.width,
                self.surface_state.config.height,
                "depth-texture",
            );
        }
    }

    pub fn reconfigure_surface(&mut self) {
        self.surface_state.reconfigure(&self.device);
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
                        r: position.x as f64 / self.surface_state.width() as f64,
                        g: position.y as f64 / self.surface_state.height() as f64,
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
                            VirtualKeyCode::M => {
                                self.current_pipeline = match self.current_pipeline {
                                    "default" | "variant" => "model",
                                    "model" => "default",
                                    _ => "default",
                                };
                                Response::Handled
                            }
                            VirtualKeyCode::T => {
                                self.current_texture_idx =
                                    (self.current_texture_idx + 1) % self.textures["binding"].len();
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

    pub fn render(&mut self, window: &Window) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface_state.surface().get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut main_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("main-render-encoder"),
                });
        {
            let depth_stencil_attachment =
                if self.current_pipeline == "model" || self.current_pipeline == "default" {
                    Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.textures["depth"][0].view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: true,
                        }),
                        stencil_ops: None,
                    })
                } else {
                    None
                };

            let mut render_pass = main_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main-render-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: true,
                    },
                })],
                depth_stencil_attachment,
            });

            render_pass.set_pipeline(&self.pipelines[self.current_pipeline]);

            match self.current_pipeline {
                "default" | "variant" => {
                    render_pass.set_bind_group(
                        0,
                        &self.bind_groups["texture"][self.current_texture_idx],
                        &[],
                    );
                    render_pass.set_bind_group(1, &self.bind_groups["camera"][0], &[]);
                    render_pass.set_bind_group(2, &self.bind_groups["depth_map"][0], &[]);
                    render_pass.set_index_buffer(self.index_buffer.data_slice(..), IDX_FORMAT);
                    render_pass.set_vertex_buffer(0, self.object_instances_buffer.slice(..));

                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        // Only bind once the vertex buffer and index buffer,
                        // then draw the different slices with vertex and index offset.
                        render_pass.set_vertex_buffer(1, self.vertex_buffer.data_slice(..));
                        self.index_buffer.subslices().iter().enumerate().for_each(
                            |(i, index_range)| {
                                let count =
                                    ((index_range.end - index_range.start) / IDX_SIZE) as u32;
                                let index_offset = (index_range.start / IDX_SIZE) as u32;
                                let vertex_offset =
                                    self.vertex_buffer.subslices()[i].start / VertexPCT::SIZE;
                                render_pass.draw_indexed(
                                    index_offset..index_offset + count,
                                    vertex_offset as i32,
                                    0..self.object_instances.len() as u32,
                                );
                            },
                        );
                    }

                    #[cfg(target_arch = "wasm32")]
                    {
                        println!("wasm32");
                        // Draw elements with base vertex is not supported using webgl.
                        self.index_buffer
                            .subslices()
                            .iter()
                            .zip(self.vertex_buffer.subslices().iter())
                            .for_each(|(index_range, vertex_range)| {
                                let vertex_buffer =
                                    self.vertex_buffer.data_slice(vertex_range.clone());
                                let index_offset = (index_range.start / IDX_SIZE) as u32;
                                let count = (index_range.end - index_range.start) / IDX_SIZE;
                                render_pass.set_vertex_buffer(1, vertex_buffer);
                                render_pass.draw_indexed(
                                    index_offset..index_offset + count as u32,
                                    0,
                                    0..self.object_instances.len() as u32,
                                );
                            });
                    }
                }

                "model" => render_pass.draw_model(
                    &self.model,
                    &self.bind_groups["camera"][0],
                    Some(&self.object_instances_buffer),
                    0..self.object_instances.len() as u32,
                ),
                _ => {}
            }
        }
        // Render GUI
        let screen_descriptor = ScreenDescriptor {
            physical_width: self.surface_state.width(),
            physical_height: self.surface_state.height(),
            scale_factor: window.scale_factor() as _,
        };

        let raw_input = self.gui_state.take_egui_input(window);
        let full_output = self.gui_ctx.run(raw_input, |ctx| {
            egui::CentralPanel::default().show(&ctx, |ui| {
                ui.label("Hello World!");
                if ui.button("Quit").clicked() {
                    println!("Quit");
                }
            });
        });
        // hanlde platform output
        let clipped_primitives = self.gui_ctx.tessellate(full_output.shapes);

        let user_cmd_buffers = {
            for (id, image_delta) in &full_output.textures_delta.set {
                self.gui_renderer
                    .update_texture(&self.device, &self.queue, *id, image_delta);
            }
            self.gui_renderer.update_buffers(
                &self.device,
                &self.queue,
                &mut main_encoder,
                &clipped_primitives,
                &screen_descriptor,
            )
        };

        let mut gui_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("main-ui-encoder"),
            });
        {
            let mut gui_render_pass = gui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main-ui-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            self.gui_renderer
                .render(&mut gui_render_pass, &clipped_primitives, screen_descriptor);
        }

        {
            for id in &full_output.textures_delta.free {
                self.gui_renderer.free_texture(*id);
            }
        }

        self.queue.submit(
            user_cmd_buffers
                .into_iter()
                .chain([main_encoder.finish(), gui_encoder.finish()].into_iter()),
        );

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
