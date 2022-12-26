use std::io::{BufReader, Cursor};

use cfg_if::cfg_if;
use wgpu::util::DeviceExt;

use crate::{
    model::{Material, Mesh, Model, VertexPTN},
    texture::Texture,
};

#[cfg(target_arch = "wasm32")]
fn format_url(filename: &str) -> reqwest::Url {
    let window = web_sys::window().unwrap();
    let location = window.location();
    let base = reqwest::Url::parse(&format!(
        "{}/{}/",
        location.origin().unwrap(),
        option_env!("RES_PATH").unwrap_or("assets"),
    ))
    .unwrap();
    base.join(filename).unwrap()
}

pub async fn load_string(filename: &str) -> anyhow::Result<String> {
    log::info!("Loading string from {}", filename);
    cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            let url = format_url(filename);
            log::info!("-- Loading from {}", url);
            let resp = reqwest::get(url).await?;
            let text = resp.text().await?;
            Ok(text)
        } else {
            let path = std::path::Path::new(env!("OUT_DIR"))
                .join("assets")
                .join(filename);
            log::info!("-- Loading from {}", path.display());
            let text = std::fs::read_to_string(path)?;
            Ok(text)
        }
    }
}

pub async fn load_binary(filename: &str) -> anyhow::Result<Vec<u8>> {
    log::info!("Loading binary from {}", filename);
    cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            let url = format_url(filename);
            log::info!("-- Loading from {}", url);
            let resp = reqwest::get(url).await?;
            let bytes = resp.bytes().await?;
            Ok(bytes.to_vec())
        } else {
            let path = std::path::Path::new(env!("OUT_DIR"))
                .join("assets")
                .join(filename);
            log::info!("-- Loading from {}", path.display());
            let bytes = std::fs::read(path)?;
            Ok(bytes)
        }
    }
}

pub async fn load_texture(
    filename: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<Texture> {
    let bytes = load_binary(filename).await?;
    Texture::from_bytes(device, queue, &bytes, filename)
}

pub async fn load_model(
    filename: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<Model> {
    log::info!("Loading model from {}", filename);
    let text = load_string(filename).await?;
    let obj_cursor = Cursor::new(text);
    let mut obj_reader = BufReader::new(obj_cursor);
    let (models, obj_materials) = tobj::load_obj_buf_async(
        &mut obj_reader,
        &tobj::LoadOptions {
            single_index: true,
            triangulate: true,
            ..Default::default()
        },
        |mat_path| async move {
            let mat_text = load_string(&mat_path).await.unwrap();
            tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
        },
    )
    .await?;

    let mut materials = Vec::new();
    for mat in obj_materials? {
        let diffuse_texture = load_texture(&mat.diffuse_texture, device, queue).await?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: None,
        });

        materials.push(Material {
            name: mat.name,
            diffuse_texture,
            bind_group,
        });
    }

    let meshes = models
        .into_iter()
        .map(|m| {
            let vertices = (0..m.mesh.positions.len() / 3)
                .map(|i| VertexPTN {
                    position: [
                        m.mesh.positions[i * 3],
                        m.mesh.positions[i * 3 + 1],
                        m.mesh.positions[i * 3 + 2],
                    ],
                    texcoord: [m.mesh.texcoords[i * 2], m.mesh.texcoords[i * 2 + 1]],
                    normal: [
                        m.mesh.normals[i * 3],
                        m.mesh.normals[i * 3 + 1],
                        m.mesh.normals[i * 3 + 2],
                    ],
                })
                .collect::<Vec<_>>();

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Vertex Buffer", filename)),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Index Buffer", filename)),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            Mesh {
                name: filename.to_string(),
                material_index: m.mesh.material_id.unwrap_or(0),
                vertex_buffer,
                index_buffer,
                num_elements: m.mesh.indices.len() as u32,
            }
        })
        .collect::<Vec<_>>();

    Ok(Model { meshes, materials })
}
