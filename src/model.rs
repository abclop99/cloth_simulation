use serde::{Deserialize, Serialize};
use wgpu::util::DeviceExt;

#[derive(Serialize, Deserialize, Debug)]
pub struct Mesh {
    pub name: String,
    pub vertices: Vec<Vertex>,
    pub springs: Vec<Spring>,
    pub triangles: Vec<Triangle>,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
    mass: f32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Spring {
    vertices: [u16; 2],
    k_s: f32,
    k_d: f32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Triangle(u16, u16, u16);

// Describes the memory layout of the vertex data
impl Vertex {
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
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
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

#[derive(Debug)]
pub struct Model {
    mesh: Mesh,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
}

impl Model {
    pub fn new(device: &wgpu::Device, mesh: Mesh) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(mesh.vertices.as_slice()),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(mesh.triangles.as_slice()),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            mesh,
            vertex_buffer,
            index_buffer,
        }
    }

    pub fn get_num_indices(&self) -> u32 {
        self.mesh.triangles.len() as u32 * 3
    }
}

pub trait DrawModel<'a> {
    fn draw_model(&mut self, model: &'a Model);
}

impl<'a, 'b> DrawModel<'a> for wgpu::RenderPass<'b>
where
    'a: 'b,
{
    fn draw_model(&mut self, model: &'b Model) {
        self.set_vertex_buffer(0, model.vertex_buffer.slice(..));
        self.set_index_buffer(model.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        self.draw_indexed(0..model.mesh.triangles.len() as u32 * 3, 0, 0..1);
    }
}
