use std::{borrow::Cow, collections::HashMap, num::NonZeroU32};

use crate::{buffer::SlicedBuffer, state::Response};

use egui::epaint;
use log::info;
use type_map::concurrent::TypeMap;
use winit::{event::WindowEvent, event_loop::EventLoopWindowTarget, window::Window};

type SetupCallback = dyn Fn(
        &wgpu::Device,
        &wgpu::Queue,
        &mut wgpu::CommandEncoder,
        &mut TypeMap,
    ) -> Vec<wgpu::CommandBuffer>
    + Send
    + Sync;

type PaintCallback = dyn for<'a, 'b> Fn(epaint::PaintCallbackInfo, &'a mut wgpu::RenderPass<'b>, &'b TypeMap)
    + Send
    + Sync;

/// A callback function that can be used to compose an [`epaint::PaintCallback`] for custom
/// WGPU rendering.
///
/// The callback is composed of two functions: `setup` and `paint`:
///
/// - `setup` is called once per frame right before `paint`, and can use the passed-in
/// [`wgpu::Device`] and [`wgpu::Buffer`] to allocate or modify GPU resources such as
/// buffers.
///
/// - `paint` is called after `setup` and is given access to the [`wgpu::RenderPass`] so
/// that it can issue draw commands into the same [`wgpu::RenderPass`] that is used for all
/// other egui elements.
///
/// The final argument of both the `setup` and `paint` functions is the
pub struct CallbackFn {
    pub setup: Box<SetupCallback>,
    pub paint: Box<PaintCallback>,
}

impl Default for CallbackFn {
    fn default() -> Self {
        Self {
            setup: Box::new(|_, _, _, _| Vec::new()),
            paint: Box::new(|_, _, _| {}),
        }
    }
}

impl CallbackFn {
    pub fn new(setup: Box<SetupCallback>, paint: Box<PaintCallback>) -> Self {
        Self { setup, paint }
    }

    /// Set the `setup` callback function.
    ///
    /// The passed-in `CommandEncoder` is egui's and can be used directly to register
    /// wgpu commands for simple use cases.
    /// This allows to reusing the same [`wgpu::CommandEncoder`] for all callbacks and
    /// egui rendering itself.
    ///
    /// For more complex use cases, one can return a list of [`wgpu::CommandBuffer`]s
    /// and have complete control over how they get created and submitted. In particular,
    /// this gives an opportunity to parallelize command registration and prevents a faulty
    /// callback from poisoning the main wgpu pipeline.
    pub fn set_setup_fn<F>(mut self, setup: F) -> Self
    where
        F: Fn(
                &wgpu::Device,
                &wgpu::Queue,
                &mut wgpu::CommandEncoder,
                &mut TypeMap,
            ) -> Vec<wgpu::CommandBuffer>
            + Send
            + Sync
            + 'static,
    {
        self.setup = Box::new(setup) as _;
        self
    }

    pub fn set_paint_fn<F>(mut self, paint: F) -> Self
    where
        F: for<'a, 'b> Fn(epaint::PaintCallbackInfo, &'a mut wgpu::RenderPass<'b>, &'b TypeMap)
            + Send
            + Sync
            + 'static,
    {
        self.paint = Box::new(paint) as _;
        self
    }
}

/// Information about the screen used for rendering.
pub struct ScreenDescriptor {
    /// The physical width of the screen in pixels.
    pub physical_width: u32,
    /// The physical height of the screen in pixels.
    pub physical_height: u32,
    /// HiDPI scale factor (pixels per point).
    pub scale_factor: f32,
}

impl ScreenDescriptor {
    /// Screen size in pixels (physical pixels).
    pub fn physical_size(&self) -> [u32; 2] {
        [self.physical_width, self.physical_height]
    }

    /// Screen size in points (logical pixels).
    pub fn logical_size(&self) -> [f32; 2] {
        [
            self.physical_width as f32 / self.scale_factor,
            self.physical_height as f32 / self.scale_factor,
        ]
    }
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct GuiUniforms {
    screen_logical_size: [f32; 2], // in points
    _padding: [u32; 2],            // Unifor buffers need to be at least 16 bytes in WebGL.
}

pub struct GuiRenderer {
    // device: Arc<wgpu::Device>,
    // queue: Arc<wgpu::Queue>,
    target_format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    index_buffer: SlicedBuffer,
    vertex_buffer: SlicedBuffer,

    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,

    texture_bind_group_layout: wgpu::BindGroupLayout,
    // Map of egui texture IDs to textures and their associated bind groups (texture + sampler)
    // The texture may be None if the TextureId is just a handle to a user-provided
    // sampler.
    textures: HashMap<egui::epaint::TextureId, (Option<wgpu::Texture>, wgpu::BindGroup)>,
    samplers: HashMap<egui::epaint::textures::TextureOptions, wgpu::Sampler>,
    next_user_texture_id: u64,
    pub paint_callback_resources: TypeMap,
}

impl GuiRenderer {
    const VERTEX_BUFFER_INIT_CAPACITY: wgpu::BufferAddress =
        1024 * std::mem::size_of::<egui::epaint::Vertex>() as wgpu::BufferAddress;

    const INDEX_BUFFER_INIT_CAPACITY: wgpu::BufferAddress =
        3 * 1024 * std::mem::size_of::<u32>() as wgpu::BufferAddress;

    pub fn new(
        device: &wgpu::Device,
        output_color_format: wgpu::TextureFormat,
        output_depth_format: Option<wgpu::TextureFormat>,
        msaa_samples: u32,
    ) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("egui_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/egui.wgsl"))),
        });
        use wgpu::util::DeviceExt;
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("egui_uniform_buffer"),
            contents: bytemuck::cast_slice(&[GuiUniforms {
                screen_logical_size: [0.0, 0.0],
                _padding: [0, 0],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("egui_uniform_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                    },
                    count: None,
                }],
            });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("egui_uniform_bind_group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("egui_texture_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("egui_pipeline_layout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });
        let depth_stencil = output_depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("egui_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<egui::epaint::Vertex>()
                        as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x2,
                        1 => Float32x2,
                        2 => Uint32,
                    ],
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
                unclipped_depth: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: if output_color_format.describe().srgb {
                    info!("Detected a linear (sRGBA aware) framebuffer {:?}. Egui prefers Rgba8Unorm or Bgra8Unorm", output_color_format);
                    "fs_main_linear_framebuffer"
                } else {
                    "fs_main_gamma_framebuffer"
                },
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_color_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            depth_stencil,
            multisample: wgpu::MultisampleState {
                count: msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        Self {
            pipeline,
            index_buffer: SlicedBuffer::new(
                device,
                Self::INDEX_BUFFER_INIT_CAPACITY,
                wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                Some("egui_index_buffer"),
            ),
            vertex_buffer: SlicedBuffer::new(
                device,
                Self::VERTEX_BUFFER_INIT_CAPACITY,
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                Some("egui_vertex_buffer"),
            ),
            uniform_buffer,
            uniform_bind_group,
            texture_bind_group_layout,
            textures: HashMap::new(),
            samplers: HashMap::new(),
            next_user_texture_id: 0,
            paint_callback_resources: TypeMap::new(),
            target_format: output_color_format,
        }
    }

    pub fn render<'rps>(
        &'rps self,
        render_pass: &mut wgpu::RenderPass<'rps>,
        primitives: &[epaint::ClippedPrimitive],
        screen_desc: &ScreenDescriptor,
    ) {
        let pixels_per_point = screen_desc.scale_factor;
        let screen_physical_size = screen_desc.physical_size();
        let mut needs_reset = true;

        let mut index_buffer_subslices = self.index_buffer.subslices().iter();
        let mut vertex_buffer_subslices = self.vertex_buffer.subslices().iter();

        for epaint::ClippedPrimitive {
            clip_rect,
            primitive,
        } in primitives
        {
            if needs_reset {
                render_pass.set_viewport(
                    0.0,
                    0.0,
                    screen_physical_size[0] as f32,
                    screen_physical_size[1] as f32,
                    0.0,
                    1.0,
                );
                render_pass.set_pipeline(&self.pipeline);
                render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                needs_reset = false;
            }

            {
                let rect = ScissorRect::new(clip_rect, pixels_per_point, screen_physical_size);

                if rect.width == 0 || rect.height == 0 {
                    // Skip rendering zero-sized clip areas.
                    if let epaint::Primitive::Mesh(_) = primitive {
                        // If this is a mesh, we need to advance the index and vertex buffer iterators.
                        index_buffer_subslices.next().unwrap();
                        vertex_buffer_subslices.next().unwrap();
                    }
                    continue;
                }

                render_pass.set_scissor_rect(rect.x, rect.y, rect.width, rect.height);
            }

            match primitive {
                epaint::Primitive::Mesh(mesh) => {
                    let index_buffer_subslice = index_buffer_subslices.next().unwrap();
                    let vertex_buffer_subslice = vertex_buffer_subslices.next().unwrap();

                    if let Some((_texture, bind_group)) = self.textures.get(&mesh.texture_id) {
                        render_pass.set_bind_group(1, bind_group, &[]);
                        render_pass.set_vertex_buffer(
                            0,
                            self.vertex_buffer
                                .data_slice(vertex_buffer_subslice.clone()),
                        );
                        render_pass.set_index_buffer(
                            self.index_buffer.data_slice(index_buffer_subslice.clone()),
                            wgpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
                    } else {
                        log::warn!("Missing texture: {:?}", mesh.texture_id);
                    }
                }
                epaint::Primitive::Callback(callback) => {
                    let Some(callback_fn) = callback.callback.downcast_ref::<CallbackFn>() else {
                        continue;
                    };

                    if callback.rect.is_positive() {
                        needs_reset = true;
                        {
                            // Set a default viewport for the render pass as a courtesy to the user,
                            // so that they don't have to think about it in the simple case where
                            // they just want to fill the whole paint area.
                            //
                            // The use still has the possibility of setting their own custom viewport
                            // during the paint callback, effectively overriding this default.
                            let min = (callback.rect.min.to_vec2() * pixels_per_point).round();
                            let max = (callback.rect.max.to_vec2() * pixels_per_point).round();

                            render_pass.set_viewport(
                                min.x,
                                min.y,
                                max.x - min.x,
                                max.y - min.y,
                                0.0,
                                1.0,
                            );
                        }
                        (callback_fn.paint)(
                            epaint::PaintCallbackInfo {
                                viewport: callback.rect,
                                clip_rect: *clip_rect,
                                pixels_per_point,
                                screen_size_px: screen_physical_size,
                            },
                            render_pass,
                            &self.paint_callback_resources,
                        );
                    }
                }
            }
        }
        render_pass.set_scissor_rect(0, 0, screen_physical_size[0], screen_physical_size[1]);
    }

    /// Uploads the given texture to the GPU and returns a [`UserTextureId`] that can be used to
    /// refer to it.
    /// Must be called before `render()`.
    pub fn update_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: epaint::TextureId,
        image_delta: &epaint::ImageDelta,
    ) {
        let width = image_delta.image.width() as u32;
        let height = image_delta.image.height() as u32;
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let data_color32 = match &image_delta.image {
            epaint::ImageData::Color(image) => {
                assert_eq!(
                    width as usize * height as usize,
                    image.pixels.len(),
                    "Image texture size and texel count mismatch"
                );
                Cow::Borrowed(&image.pixels)
            }
            epaint::ImageData::Font(image) => {
                assert_eq!(
                    width as usize * height as usize,
                    image.pixels.len(),
                    "Image texture size and texel count mismatch"
                );
                Cow::Owned(image.srgba_pixels(None).collect::<Vec<_>>())
            }
        };
        let data_bytes: &[u8] = bytemuck::cast_slice(data_color32.as_slice());
        let queue_write_data_to_texture = |texture, origin| {
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture,
                    mip_level: 0,
                    origin,
                    aspect: wgpu::TextureAspect::All,
                },
                data_bytes,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(4 * width),
                    rows_per_image: NonZeroU32::new(height),
                },
                size,
            );
        };
        if let Some(pos) = image_delta.pos {
            // Update existing texture:
            let (texture, _) = self
                .textures
                .get(&id)
                .expect("Tried to update a texture that has not been allocated yet.");
            let origin = wgpu::Origin3d {
                x: pos[0] as u32,
                y: pos[1] as u32,
                z: 0,
            };
            queue_write_data_to_texture(
                texture.as_ref().expect("Tried to update user texture."),
                origin,
            );
        } else {
            // Allocate a new texture use the same label for all resources associated with this texture id
            let label_str = format!("egui_tex_id_{:?}", id);
            let label = Some(label_str.as_str());
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb, // minimal emulation wgpu WebGL emulation is WebGL2, so this should always be supported
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            });
            let sampler = self
                .samplers
                .entry(image_delta.options)
                .or_insert_with(|| create_sampler(device, image_delta.options));
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label,
                layout: &self.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &texture.create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            let origin = wgpu::Origin3d::ZERO;
            queue_write_data_to_texture(&texture, origin);
            self.textures.insert(id, (Some(texture), bind_group));
        }
    }

    pub fn free_texture(&mut self, id: epaint::TextureId) {
        self.textures.remove(&id);
    }

    /// Get the WGPU texture and bind group associated to a texture that has been allocated by egui.
    ///
    /// This could be used by custom paint hooks to render images that have been added through with
    /// [`egui_extras::RetainedImage`] or [`epaint::Context::load_texture`].
    pub fn texture(
        &self,
        id: epaint::TextureId,
    ) -> Option<&(Option<wgpu::Texture>, wgpu::BindGroup)> {
        self.textures.get(&id)
    }

    /// Registers a `wgpu::Texture` with a `epaint::TextureId`.
    ///
    /// This enables the application to reference the texture inside an image ui element.
    /// This effectively enables offscreen rendering inside the egui UI. Texture must have
    /// the texture format `wgpu::TextureFormat::Rgba8UnormSrgb` and texture usage `TextureUsage::SAMPLED`.
    pub fn register_native_texture(
        &mut self,
        device: &wgpu::Device,
        texture: &wgpu::TextureView,
        texture_filter: wgpu::FilterMode,
    ) -> epaint::TextureId {
        self.register_native_texture_with_sampler_options(
            device,
            texture,
            wgpu::SamplerDescriptor {
                label: Some(format!("egui_tex_user_id_{}", self.next_user_texture_id).as_str()),
                mag_filter: texture_filter,
                min_filter: texture_filter,
                ..Default::default()
            },
        )
    }

    /// Registers a `wgpu::Texture` with a `epaint::TextureId` while also accepting custom `wgpu::SamplerDescriptor` options.
    ///
    /// This allows applications to specify individual minification and magnification filters as well as
    /// custom mipmapping tiling options.
    ///
    /// The `Texture` must have the texture format `wgpu::TextureFormat::Rgba8UnormSrgb` and texture usage `TextureUsage::SAMPLED`.
    /// Any compare function specified in the `wgpu::SamplerDescriptor` will be ignored.
    pub fn register_native_texture_with_sampler_options(
        &mut self,
        device: &wgpu::Device,
        texture: &wgpu::TextureView,
        sampler: wgpu::SamplerDescriptor<'_>,
    ) -> epaint::TextureId {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            compare: None,
            ..sampler
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(format!("egui_tex_user_id_{}", self.next_user_texture_id).as_str()),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let id = epaint::TextureId::User(self.next_user_texture_id);
        self.textures.insert(id, (None, bind_group));
        self.next_user_texture_id += 1;
        id
    }

    /// Registers a `wgpu::Texture` with an existing `epaint::TextureId`.
    ///
    /// This enables applications to resuse an existing `epaint::TextureId` with a new `wgpu::TextureView`.
    pub fn update_egui_texture_from_wgpu_texture(
        &mut self,
        device: &wgpu::Device,
        texture: &wgpu::TextureView,
        texture_filter: wgpu::FilterMode,
        id: epaint::TextureId,
    ) {
        self.update_egui_texture_from_wgpu_texture_with_sampler_options(
            device,
            texture,
            wgpu::SamplerDescriptor {
                label: Some(format!("egui_tex_user_id_{}", self.next_user_texture_id).as_str()),
                mag_filter: texture_filter,
                min_filter: texture_filter,
                ..Default::default()
            },
            id,
        )
    }

    /// Registers a `wgpu::Texture` with a `epaint::TextureId` while also accepting custom `wgpu::SamplerDescriptor`
    /// This allows applications to reuse an existing `epaint::TextureId` with a new `wgpu::TextureView` and custom `wgpu::SamplerDescriptor` options.
    pub fn update_egui_texture_from_wgpu_texture_with_sampler_options(
        &mut self,
        device: &wgpu::Device,
        texture: &wgpu::TextureView,
        sampler: wgpu::SamplerDescriptor<'_>,
        id: epaint::TextureId,
    ) {
        let (_user_texture, user_texture_bind_group) = self
            .textures
            .get_mut(&id)
            .expect("Tried to update a textue that has not been allocated yet.");

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            compare: None,
            ..sampler
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(format!("egui_tex_user_id_{}", self.next_user_texture_id).as_str()),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        *user_texture_bind_group = bind_group;
    }

    /// Uploads the uniforms, vertex and index data used by the renderer.
    /// Should be called before [`Self::render`].
    /// Returns all user-defined command buffers gathered from setup callbacks.
    pub fn update_buffers(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        primitives: &[epaint::ClippedPrimitive],
        screen_desc: &ScreenDescriptor,
    ) -> Vec<wgpu::CommandBuffer> {
        let screen_size_in_points = screen_desc.logical_size();

        {
            queue.write_buffer(
                &self.uniform_buffer,
                0,
                bytemuck::cast_slice(&[GuiUniforms {
                    screen_logical_size: screen_size_in_points,
                    _padding: [0; 2],
                }]),
            );
        }

        // Determine how many vertices & indices need to be rendered.
        let (vertex_count, index_cout) =
            primitives.iter().fold((0, 0), |acc, clipped_primitive| {
                match &clipped_primitive.primitive {
                    epaint::Primitive::Mesh(mesh) => {
                        (acc.0 + mesh.vertices.len(), acc.1 + mesh.indices.len())
                    }
                    epaint::Primitive::Callback(_) => acc,
                }
            });

        // Resize the vertex and index buffers if needed.
        {
            self.index_buffer.subslices_mut().clear();
            let required_size = (std::mem::size_of::<u32>() * index_cout) as wgpu::BufferAddress;
            if self.index_buffer.capacity() < required_size {
                self.index_buffer.grow(device, queue, required_size, false);
            }
        }
        {
            self.vertex_buffer.subslices_mut().clear();
            let required_size =
                (std::mem::size_of::<epaint::Vertex>() * vertex_count) as wgpu::BufferAddress;
            if self.vertex_buffer.capacity() < required_size {
                self.vertex_buffer.grow(device, queue, required_size, false);
            }
        }

        // Upload index and vertex data and call user-defined setup callbacks.
        let mut user_command_buffers = Vec::new(); // collect user command buffers
        for epaint::ClippedPrimitive { primitive, .. } in primitives.iter() {
            match primitive {
                epaint::Primitive::Mesh(mesh) => {
                    {
                        // Upload index data.
                        let index_offset =
                            self.index_buffer.subslices().last().unwrap_or(&(0..0)).end;
                        let data = bytemuck::cast_slice(&mesh.indices);
                        queue.write_buffer(&self.index_buffer.buffer(), index_offset, data);
                        self.index_buffer
                            .subslices_mut()
                            .push(index_offset..index_offset + data.len() as wgpu::BufferAddress);
                    }
                    {
                        // Upload vertex data.
                        let vertex_offset =
                            self.vertex_buffer.subslices().last().unwrap_or(&(0..0)).end;
                        let data = bytemuck::cast_slice(&mesh.vertices);
                        queue.write_buffer(&self.vertex_buffer.buffer(), vertex_offset, data);
                        self.vertex_buffer
                            .subslices_mut()
                            .push(vertex_offset..vertex_offset + data.len() as wgpu::BufferAddress);
                    }
                }
                epaint::Primitive::Callback(callback) => {
                    let Some(callback) = callback.callback.downcast_ref::<CallbackFn>() else {
                        log::warn!("Unknown paint callback function: expected `CallbackFn`");
                        continue;
                    };
                    user_command_buffers.extend((callback.setup)(
                        device,
                        queue,
                        encoder,
                        &mut self.paint_callback_resources,
                    ));
                }
            }
        }
        user_command_buffers
    }
}

/// A Rect in physical pixels.
struct ScissorRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl ScissorRect {
    fn new(clip_rect: &egui::Rect, pixels_per_point: f32, target_size: [u32; 2]) -> Self {
        // Transform clip rect to physical pixels:
        let clip_min_x = clip_rect.min.x * pixels_per_point;
        let clip_min_y = clip_rect.min.y * pixels_per_point;
        let clip_max_x = clip_rect.max.x * pixels_per_point;
        let clip_max_y = clip_rect.max.y * pixels_per_point;

        // Round to integer pixels:
        let clip_min_x = clip_min_x.round() as u32;
        let clip_min_y = clip_min_y.round() as u32;
        let clip_max_x = clip_max_x.round() as u32;
        let clip_max_y = clip_max_y.round() as u32;

        // Clamp to screen:
        let clip_min_x = clip_min_x.clamp(0, target_size[0]);
        let clip_min_y = clip_min_y.clamp(0, target_size[1]);
        let clip_max_x = clip_max_x.clamp(clip_min_x, target_size[0]);
        let clip_max_y = clip_max_y.clamp(clip_min_y, target_size[1]);

        Self {
            x: clip_min_x,
            y: clip_min_y,
            width: clip_max_x - clip_min_x,
            height: clip_max_y - clip_min_y,
        }
    }
}

fn create_sampler(
    device: &wgpu::Device,
    options: epaint::textures::TextureOptions,
) -> wgpu::Sampler {
    let mag_filter = match options.magnification {
        egui::TextureFilter::Nearest => wgpu::FilterMode::Nearest,
        egui::TextureFilter::Linear => wgpu::FilterMode::Linear,
    };
    let min_filter = match options.minification {
        egui::TextureFilter::Nearest => wgpu::FilterMode::Nearest,
        egui::TextureFilter::Linear => wgpu::FilterMode::Linear,
    };
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(&format!(
            "egui_sampler_(mag: {:?}, min: {:?})",
            mag_filter, min_filter
        )),
        mag_filter,
        min_filter,
        ..Default::default()
    })
}

pub struct GuiContext {
    egui_context: egui::Context,
    egui_state: egui_winit::State,
    egui_input: egui::RawInput,
}

impl GuiContext {
    pub fn handle_event(&mut self, event: &WindowEvent) -> Response {
        match self.egui_state.on_event(&self.egui_context, event).consumed {
            true => Response::Handled,
            false => Response::Ignored,
        }
    }
}

impl GuiContext {
    pub fn new(event_loop: &EventLoopWindowTarget<()>) -> Self {
        Self {
            egui_context: egui::Context::default(),
            egui_state: egui_winit::State::new(event_loop),
            egui_input: egui::RawInput::default(),
        }
    }

    pub fn take_input(&mut self, window: &Window) {
        self.egui_input = self.egui_state.take_egui_input(window);
    }

    pub fn handle_platform_output(&mut self, window: &Window, output: egui::PlatformOutput) {
        self.egui_state
            .handle_platform_output(window, &self.egui_context, output);
    }

    pub fn run(&mut self, ui: impl FnOnce(&egui::Context)) -> egui::FullOutput {
        self.egui_context.run(self.egui_input.take(), ui)
    }
}

pub struct GuiState {
    pub context: GuiContext,
    pub renderer: GuiRenderer,
}

impl GuiState {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        event_loop: &EventLoopWindowTarget<()>,
    ) -> Self {
        let context = GuiContext::new(event_loop);
        let renderer = GuiRenderer::new(device, target_format, None, 1);
        Self { context, renderer }
    }

    pub fn update(&mut self, window: &Window) {
        self.context.take_input(window);
    }

    /// Returns user command buffers and ui rendering command buffer.
    pub fn render(
        &mut self,
        window: &Window,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_desc: ScreenDescriptor,
        target: &wgpu::TextureView,
        ui: impl FnOnce(&egui::Context),
    ) -> (Vec<wgpu::CommandBuffer>, wgpu::CommandBuffer) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gui-encoder"),
        });
        let output = self.context.run(ui);
        self.context
            .handle_platform_output(&window, output.platform_output);

        let primitives = self.context.egui_context.tessellate(output.shapes);

        let user_command_buffers = {
            for (id, image_delta) in &output.textures_delta.set {
                self.renderer
                    .update_texture(device, queue, *id, image_delta);
            }
            self.renderer
                .update_buffers(device, queue, &mut encoder, &primitives, &screen_desc)
        };

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gui_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            self.renderer
                .render(&mut render_pass, &primitives, &screen_desc);
        }

        {
            for id in &output.textures_delta.free {
                self.renderer.free_texture(*id);
            }
        }

        (user_command_buffers, encoder.finish())
    }
}
