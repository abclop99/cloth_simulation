use cgmath::prelude::*;
use instant::Duration;
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
    #[serde(default)]
    normal: [f32; 3],
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
                    offset: std::mem::size_of::<[f32; 3 * 1]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3 * 2]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3 * 3]>() as wgpu::BufferAddress,
                    shader_location: 3,
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
    pub fn new(device: &wgpu::Device, mut mesh: Mesh) -> Self {
        // Calculate the normals for each vertex and set them in the mesh
        let triangle_normals = Self::compute_triangle_normals(&mesh.vertices, &mesh.triangles);
        let vertex_normals: Vec<[f32; 3]> =
            Self::compute_vertex_normals(&mesh.vertices, &mesh.triangles, &triangle_normals);

        for (vertex, normal) in mesh.vertices.iter_mut().zip(vertex_normals) {
            vertex.normal = normal;
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(mesh.vertices.as_slice()),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
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

    pub fn update(&mut self, _timestep: Duration, queue: &wgpu::Queue) {
        let triangle_normals =
            Self::compute_triangle_normals(&self.mesh.vertices, &self.mesh.triangles);
        let vertex_normals = Self::compute_vertex_normals(
            &self.mesh.vertices,
            &self.mesh.triangles,
            &triangle_normals,
        );

        for (vertex, normal) in self.mesh.vertices.iter_mut().zip(vertex_normals) {
            vertex.normal = normal;
        }

        queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(self.mesh.vertices.as_slice()),
        );
    }

    fn get_num_indices(&self) -> u32 {
        self.mesh.triangles.len() as u32 * 3
    }

    // Compute the normals of each triangle, but not normalized so averaging
    // them for each vertex will give a better result
    fn compute_triangle_normals(
        vertices: &Vec<Vertex>,
        triangles: &Vec<Triangle>,
    ) -> Vec<cgmath::Vector3<f32>> {
        triangles
            .iter()
            .map(|Triangle(v1, v2, v3)| {
                let v1: cgmath::Point3<f32> = vertices[*v1 as usize].position.into();
                let v2: cgmath::Point3<f32> = vertices[*v2 as usize].position.into();
                let v3: cgmath::Point3<f32> = vertices[*v3 as usize].position.into();

                (v2 - v1).cross(v3 - v1)
            })
            .collect()
    }

    fn compute_vertex_normals(
        vertices: &Vec<Vertex>,
        triangles: &Vec<Triangle>,
        triangle_normals: &Vec<cgmath::Vector3<f32>>,
    ) -> Vec<[f32; 3]> {
        let mut vertex_normals: Vec<cgmath::Vector3<f32>> =
            vec![(0.0, 0.0, 0.0).into(); vertices.len()];

        for (triangle, normal) in triangles.iter().zip(triangle_normals) {
            vertex_normals[triangle.0 as usize] += *normal;
            vertex_normals[triangle.1 as usize] += *normal;
            vertex_normals[triangle.2 as usize] += *normal;
        }

        vertex_normals
            .iter()
            .map(|normal| normal.normalize().into())
            .collect::<Vec<_>>()
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
        self.draw_indexed(0..model.get_num_indices(), 0, 0..1);
    }
}
