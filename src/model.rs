use cgmath::prelude::*;
use instant::Duration;
use serde::{Deserialize, Serialize};
use wgpu::util::DeviceExt;
use winit::event::*;

#[derive(Serialize, Deserialize, Debug)]
pub struct Mesh {
    pub name: String,
    pub settings: SimulationSettings,
    pub vertices: Vec<Vertex>,
    pub springs: Vec<Spring>,
    pub triangles: Vec<Triangle>,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
    #[serde(default)]
    pub normal: [f32; 3],
    pub mass: f32,
    pub fixed: i32, // Typed as i32 because Pod and transmute?
    #[serde(default)]
    pub velocity: [f32; 3],
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct Spring {
    pub vertices: [u16; 2],
    pub k_s: f32,
    pub k_d: f32,
    #[serde(default)]
    pub rest_length: f32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Triangle(pub u16, pub u16, pub u16);

impl Triangle {
    pub fn normal(&self, vertices: &[Vertex]) -> cgmath::Vector3<f32> {
        let v0 = cgmath::Vector3::from(vertices[self.0 as usize].position);
        let v1 = cgmath::Vector3::from(vertices[self.1 as usize].position);
        let v2 = cgmath::Vector3::from(vertices[self.2 as usize].position);

        let e0 = v1 - v0;
        let e1 = v2 - v0;

        e0.cross(e1).normalize()
    }

    pub fn area(&self, vertices: &[Vertex]) -> f32 {
        let v0 = cgmath::Vector3::from(vertices[self.0 as usize].position);
        let v1 = cgmath::Vector3::from(vertices[self.1 as usize].position);
        let v2 = cgmath::Vector3::from(vertices[self.2 as usize].position);

        let e0 = v1 - v0;
        let e1 = v2 - v0;

        0.5 * e0.cross(e1).magnitude()
    }
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct SimulationSettings {
    pub gravity: [f32; 3],
    pub wind: [f32; 3],
    pub fluid_density: f32,
    pub ground_level: f32,
    pub ground_size: f32,
    pub ground_friction_static: f32,
    pub ground_friction_dynamic: f32,
    pub ground_restitution: f32,
    pub ground_color: [f32; 3],
}

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
pub struct SimulationModel {
    mesh: Mesh,

    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,

    triangle_normals: Vec<cgmath::Vector3<f32>>,

    // Fixed vertices that can be moved by the user
    movable_vertices: Vec<usize>,

    paused: bool,
    mouse_pressed: bool,
}

impl SimulationModel {
    pub fn new(device: &wgpu::Device, mut mesh: Mesh) -> Self {
        // Set fixed, user movable vertices before adding ground plane
        let movable_vertices = mesh
            .vertices
            .iter()
            .enumerate()
            .filter(|(_, v)| v.fixed == 1)
            .map(|(i, _)| i)
            .collect();

        Self::add_ground_plane(&mut mesh);

        // Calculate the normals for each vertex and set them in the mesh
        let triangle_normals = Self::compute_triangle_normals(&mesh.vertices, &mesh.triangles);
        let vertex_normals: Vec<[f32; 3]> =
            Self::compute_vertex_normals(&mesh.vertices, &mesh.triangles, &triangle_normals);

        for (vertex, normal) in mesh.vertices.iter_mut().zip(vertex_normals) {
            vertex.normal = normal;
        }

        // Create the vertex and index buffers
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

        // Set spring rest lengths to initial distance between vertices
        for spring in mesh.springs.iter_mut() {
            let v1 = mesh.vertices[spring.vertices[0] as usize].position;
            let v2 = mesh.vertices[spring.vertices[1] as usize].position;
            spring.rest_length = cgmath::Vector3::from(v1).distance(cgmath::Vector3::from(v2));
        }

        // Set initial velocities
        for vertex in mesh.vertices.iter_mut() {
            vertex.velocity = [0.0; 3];
        }

        Self {
            mesh,
            vertex_buffer,
            index_buffer,
            triangle_normals,
            movable_vertices,
            paused: false,
            mouse_pressed: false,
        }
    }

    // Add ground plane as fixed vertices and traingles added to the mesh
    fn add_ground_plane(mesh: &mut Mesh) {
        let ground_level = mesh.settings.ground_level;
        let ground_size = mesh.settings.ground_size;

        let mut new_vertices: Vec<Vertex> = vec![];

        for x in 0..2 {
            for z in 0..2 {
                let x = (x as f32 - 0.5) * ground_size;
                let z = (z as f32 - 0.5) * ground_size;

                new_vertices.push(Vertex {
                    position: [x, ground_level, z],
                    color: mesh.settings.ground_color,
                    normal: [0.0, 1.0, 0.0],
                    mass: 0.0,
                    fixed: 1,
                    velocity: [0.0; 3],
                });
            }
        }

        // Add triangle indices
        let first_vertex_index = mesh.vertices.len() as u16;
        mesh.triangles.push(Triangle(
            first_vertex_index,
            first_vertex_index + 1,
            first_vertex_index + 2,
        ));
        mesh.triangles.push(Triangle(
            first_vertex_index + 2,
            first_vertex_index + 1,
            first_vertex_index + 3,
        ));

        mesh.vertices.append(&mut new_vertices);
    }

    pub fn process_window_event(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput { input, .. } => {
                if let Some(keycode) = input.virtual_keycode {
                    match keycode {
                        VirtualKeyCode::Space => {
                            if input.state == ElementState::Pressed {
                                self.paused = !self.paused;
                                true
                            } else {
                                false
                            }
                        }
                        _ => false,
                    }
                } else {
                    false
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Left {
                    self.mouse_pressed = *state == ElementState::Pressed;
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    pub fn input_mouse_motion(
        &mut self,
        mouse_delta: (f64, f64),
        camera_view: cgmath::Matrix4<f32>,
        screen_size: &winit::dpi::PhysicalSize<u32>,
    ) {
        if self.mouse_pressed {
            const MOUSE_SENSITIVITY: f32 = 12.0;

            let inv_camera_view = match camera_view.invert() {
                Some(inv) => inv,
                None => return,
            };

            // Move the movable vertices
            let vertex_delta: cgmath::Vector3<f32> =
                [mouse_delta.0 as f32, -mouse_delta.1 as f32, 0.0].into();
            let vertex_delta = inv_camera_view.transform_vector(vertex_delta);

            for vertex_index in self.movable_vertices.iter() {
                self.mesh.vertices[*vertex_index].position[0] +=
                    vertex_delta.x / screen_size.width as f32 * MOUSE_SENSITIVITY;
                self.mesh.vertices[*vertex_index].position[1] +=
                    vertex_delta.y / screen_size.width as f32 * MOUSE_SENSITIVITY;
                self.mesh.vertices[*vertex_index].position[2] +=
                    vertex_delta.z / screen_size.width as f32 * MOUSE_SENSITIVITY;
            }
        }
    }

    pub fn update(&mut self, timestep: Duration, queue: &wgpu::Queue) {
        if self.paused {
            return;
        }

        let mut forces: Vec<cgmath::Vector3<f32>> = vec![[0.0; 3].into(); self.mesh.vertices.len()];

        Self::apply_gravity(&mut forces, &self.mesh.vertices, self.mesh.settings.gravity);
        Self::apply_spring_damper_forces(&self.mesh.vertices, &self.mesh.springs, &mut forces);
        Self::apply_aerodynamic_forces(
            &self.mesh.vertices,
            &self.mesh.triangles,
            &self.mesh.settings,
            &mut forces,
        );

        //Self::limit_lengths(&self.mesh.springs, &self.mesh.vertices, &mut forces);

        self.integrate_forces(&forces, timestep);

        self.update_normals(queue);
    }

    fn apply_gravity(
        forces: &mut Vec<cgmath::Vector3<f32>>,
        vertices: &Vec<Vertex>,
        gravity: [f32; 3],
    ) {
        for (force, vertex) in forces.iter_mut().zip(vertices.iter()) {
            let gravity: cgmath::Vector3<f32> = gravity.into();
            *force += gravity * vertex.mass;
        }
    }

    fn apply_spring_damper_forces(
        vertices: &Vec<Vertex>,
        springs: &Vec<Spring>,
        forces: &mut Vec<cgmath::Vector3<f32>>,
    ) {
        for spring in springs {
            let i_1 = spring.vertices[0] as usize;
            let i_2 = spring.vertices[1] as usize;

            let r_1: cgmath::Point3<f32> = vertices[i_1].position.into();
            let r_2: cgmath::Point3<f32> = vertices[i_2].position.into();

            let e = (r_2 - r_1).normalize();

            // Spring
            // F_s = -k_s * (l - l_0) * e
            // l = distance between vertices
            // l_0 = rest length
            // e = normalized vector from v1 to v2
            let l = r_1.distance(r_2);
            let x = spring.rest_length - l;
            let f_s = -spring.k_s * x * e;

            // Damper
            // F_d = -k_d * (v_close) * e
            // v_close = (v1 - v2) * e
            let v_1: cgmath::Vector3<f32> = vertices[i_1].velocity.into();
            let v_2: cgmath::Vector3<f32> = vertices[i_2].velocity.into();
            let v_close = (v_1 - v_2).dot(e);
            let f_d = -spring.k_d * v_close * e;

            // Limit forces to finite values
            // This is necessary because the simulation can get unstable
            // and produce NaN values
            if f_s.magnitude2().is_finite() && f_d.magnitude2().is_finite() {
                forces[i_1] += f_s + f_d;
                forces[i_2] -= f_s + f_d;
            }
        }
    }

    fn apply_aerodynamic_forces(
        vertices: &Vec<Vertex>,
        triangles: &Vec<Triangle>,
        settings: &SimulationSettings,
        forces: &mut Vec<cgmath::Vector3<f32>>,
    ) {
        const COEFFICIENT: f32 = 1.28;

        for triangle in triangles {
            let i_1 = triangle.0 as usize;
            let i_2 = triangle.1 as usize;
            let i_3 = triangle.2 as usize;

            let r_1: cgmath::Point3<f32> = vertices[i_1].position.into();
            let r_2: cgmath::Point3<f32> = vertices[i_2].position.into();
            let r_3: cgmath::Point3<f32> = vertices[i_3].position.into();

            let e_1 = (r_2 - r_1).normalize();
            let e_2 = (r_3 - r_1).normalize();

            let n = e_1.cross(e_2).normalize();

            // Aerodynamic
            // F_a = -0.5 * density * (v)^2 * coefficient * area * n
            let v_1: cgmath::Vector3<f32> = vertices[i_1].velocity.into();
            let v_2: cgmath::Vector3<f32> = vertices[i_2].velocity.into();
            let v_3: cgmath::Vector3<f32> = vertices[i_3].velocity.into();
            let v: cgmath::Vector3<f32> =
                ((v_1 + v_2 + v_3) / 3.0) - cgmath::Vector3::from(settings.wind);

            let density = settings.fluid_density;

            let area = triangle.area(vertices);

            let f_a = -0.5 * density * v.magnitude2() * COEFFICIENT * area * n;

            // Add forces to vertices
            if f_a.magnitude2().is_finite() {
                forces[i_1] += f_a * 0.333;
                forces[i_2] += f_a * 0.333;
                forces[i_3] += f_a * 0.333;
            }
        }
    }

    // Removes the forces if the spring is too long and the force is pulling the vertices apart
    fn limit_lengths(
        springs: &Vec<Spring>,
        vertices: &Vec<Vertex>,
        forces: &mut Vec<cgmath::Vector3<f32>>,
    ) {
        // Apply more dampening force if the spring is too long and moving apart and
        // the force is pulling the vertices further apart
        for spring in springs {
            let i_1 = spring.vertices[0] as usize;
            let i_2 = spring.vertices[1] as usize;

            let r_1: cgmath::Point3<f32> = vertices[i_1].position.into();
            let r_2: cgmath::Point3<f32> = vertices[i_2].position.into();

            let e = (r_2 - r_1).normalize();

            let l = r_1.distance(r_2);

            if l < spring.rest_length * 1.5 {
                continue;
            }

            // Damper
            // F_d = -k_d * (v_close) * e
            // v_close = (v1 - v2) * e
            let v_1: cgmath::Vector3<f32> = vertices[i_1].velocity.into();
            let v_2: cgmath::Vector3<f32> = vertices[i_2].velocity.into();
            let v_close = (v_1 - v_2).dot(e);

            if v_close >= 0.0 {
                continue;
            }

            let f_1 = forces[i_1];
            let f_2 = forces[i_2];
            let f_close = (f_1 - f_2).dot(e);

            if f_close >= 0.0 {
                continue;
            }

            let f_d = -0.0 * v_close * e;

            // Limit forces to finite values
            // This is necessary because the simulation can get unstable
            // and produce NaN values
            if f_d.magnitude2().is_finite() {
                forces[i_1] += f_d;
                forces[i_2] -= f_d;
            }
        }
    }

    fn integrate_forces(&mut self, forces: &Vec<cgmath::Vector3<f32>>, timestep: Duration) {
        // Integrate forces
        for (vertex, force) in self.mesh.vertices.iter_mut().zip(forces) {
            let acceleration = *force / vertex.mass;

            if vertex.fixed != 0 {
                vertex.velocity = [0.0; 3];
                continue;
            } else {
                if acceleration.magnitude2().is_finite() {
                    vertex.velocity[0] += acceleration.x * timestep.as_secs_f32();
                    vertex.velocity[1] += acceleration.y * timestep.as_secs_f32();
                    vertex.velocity[2] += acceleration.z * timestep.as_secs_f32();
                }

                // Clip velocity magnitude to limit the amount of instability
                if cgmath::Vector3::from(vertex.velocity).magnitude2() > 10000.0 {
                    vertex.velocity =
                        (cgmath::Vector3::from(vertex.velocity).normalize() * 10.0).into();
                }

                let velocity: cgmath::Vector3<f32> = vertex.velocity.into();
                let new_position: cgmath::Point3<f32> = vertex.position.into();
                let new_postion = new_position + velocity * timestep.as_secs_f32();

                Self::ground_collistion(&self.mesh.settings, vertex, force, &new_postion, timestep);
            }
        }
    }

    fn ground_collistion(
        settings: &SimulationSettings,
        vertex: &mut Vertex,
        force: &cgmath::Vector3<f32>,
        new_position: &cgmath::Point3<f32>,
        timestep: Duration,
    ) {
        const GROUND_LEVEL_EPSILON: f32 = 0.001;
        let _ground_level = settings.ground_level + GROUND_LEVEL_EPSILON;

        // Threshold of movement before friction is applied
        const STATIC_FRICTION_THRESHOLD: f32 = 0.01;

        if vertex.fixed == 0 && new_position[1] < _ground_level {
            // Friction
            let velocity: cgmath::Vector3<f32> = vertex.velocity.into();
            let v_tan =
                velocity - velocity.dot(cgmath::Vector3::unit_y()) * cgmath::Vector3::unit_y();
            let f_n = force.dot(cgmath::Vector3::unit_y()) * cgmath::Vector3::unit_y();

            let f_s = (-f_n * settings.ground_friction_static).magnitude() * v_tan.normalize();

            let impulse: cgmath::Vector3<f32> = if v_tan.magnitude() < STATIC_FRICTION_THRESHOLD
                && f_s.magnitude2() / vertex.mass < v_tan.magnitude2()
            {
                -v_tan
            } else {
                (-f_n * settings.ground_friction_dynamic).magnitude()
                    * (-v_tan.normalize())
                    * timestep.as_secs_f32()
                    / vertex.mass
            };

            if impulse.magnitude2().is_finite() {
                vertex.velocity[0] += impulse.x;
                vertex.velocity[1] += impulse.y;
                vertex.velocity[2] += impulse.z;
            }

            // collision
            let penetration = _ground_level - new_position[1];
            vertex.position[1] = _ground_level + settings.ground_restitution * penetration;
            vertex.velocity[1] = -vertex.velocity[1] * settings.ground_restitution;

            vertex.position[0] = new_position[0];
            //vertex.position[1] = new_position[1];
            vertex.position[2] = new_position[2];
        } else {
            vertex.position[0] = new_position[0];
            vertex.position[1] = new_position[1];
            vertex.position[2] = new_position[2];
        }
    }

    fn update_normals(&mut self, queue: &wgpu::Queue) {
        self.triangle_normals =
            Self::compute_triangle_normals(&self.mesh.vertices, &self.mesh.triangles);
        let vertex_normals = Self::compute_vertex_normals(
            &self.mesh.vertices,
            &self.mesh.triangles,
            &self.triangle_normals,
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
            .map(|triangle| triangle.normal(vertices))
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
    fn draw_model(&mut self, model: &'a SimulationModel);
}

impl<'a, 'b> DrawModel<'a> for wgpu::RenderPass<'b>
where
    'a: 'b,
{
    fn draw_model(&mut self, model: &'b SimulationModel) {
        self.set_vertex_buffer(0, model.vertex_buffer.slice(..));
        self.set_index_buffer(model.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        self.draw_indexed(0..model.get_num_indices(), 0, 0..1);
    }
}
