use cgmath::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::mpsc::channel;
use std::time::Duration;
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
    left_mouse_pressed: bool,
    wind_adjust_pressed: bool,
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

        mesh.vertices
            .par_iter_mut()
            .zip(vertex_normals)
            .for_each(|(vertex, normal)| {
                vertex.normal = normal;
            });

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
        mesh.springs.par_iter_mut().for_each(|spring| {
            let v1 = mesh.vertices[spring.vertices[0] as usize].position;
            let v2 = mesh.vertices[spring.vertices[1] as usize].position;
            spring.rest_length = cgmath::Vector3::from(v1).distance(cgmath::Vector3::from(v2));
        });

        // Set initial velocities
        mesh.vertices.par_iter_mut().for_each(|vertex| {
            vertex.velocity = [0.0; 3];
        });

        Self {
            mesh,
            vertex_buffer,
            index_buffer,
            triangle_normals,
            movable_vertices,
            paused: false,
            left_mouse_pressed: false,
            wind_adjust_pressed: false,
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
                        VirtualKeyCode::W => {
                            self.wind_adjust_pressed = input.state == ElementState::Pressed;

                            if !self.wind_adjust_pressed {
                                println!("Wind speed: {:?}", self.mesh.settings.wind);
                            }
                            true
                        }
                        _ => false,
                    }
                } else {
                    false
                }
            }
            WindowEvent::MouseInput { state, button, .. } => match button {
                MouseButton::Left => {
                    self.left_mouse_pressed = *state == ElementState::Pressed;
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    pub fn input_mouse_motion(
        &mut self,
        mouse_delta: (f64, f64),
        camera_view: cgmath::Matrix4<f32>,
        screen_size: &winit::dpi::PhysicalSize<u32>,
    ) {
        if self.wind_adjust_pressed {
            const WIND_ADJUST_SENSITIVITY: f32 = 5.0;

            let screen_height = screen_size.height as f32;

            let delta_x: f32 = mouse_delta.0 as f32 * WIND_ADJUST_SENSITIVITY / screen_height;
            let delta_y: f32 = mouse_delta.1 as f32 * WIND_ADJUST_SENSITIVITY / screen_height;

            self.mesh.settings.wind[0] += delta_x;
            self.mesh.settings.wind[2] += delta_y;
        } else if self.left_mouse_pressed {
            const MOUSE_SENSITIVITY: f32 = 20.0;

            let inv_camera_view = match camera_view.invert() {
                Some(inv) => inv,
                None => return,
            };

            // Move the movable vertices
            let vertex_delta: cgmath::Vector3<f32> =
                [mouse_delta.0 as f32, -mouse_delta.1 as f32, 0.0].into();
            let vertex_delta = inv_camera_view.transform_vector(vertex_delta);

            let screen_height = screen_size.height as f32;

            for vertex_index in self.movable_vertices.iter() {
                self.mesh.vertices[*vertex_index].position[0] +=
                    vertex_delta.x / screen_height * MOUSE_SENSITIVITY;
                self.mesh.vertices[*vertex_index].position[1] +=
                    vertex_delta.y / screen_height * MOUSE_SENSITIVITY;
                self.mesh.vertices[*vertex_index].position[2] +=
                    vertex_delta.z / screen_height * MOUSE_SENSITIVITY;
            }
        }
    }

    pub fn update(&mut self, timestep: Duration, queue: &wgpu::Queue, depth: i32) {
        if self.paused {
            return;
        }

        // Recursively split timestep if too long
        if timestep.as_secs_f32() > 0.016 && depth > 0 {
            for _ in 0..2 {
                self.update(timestep.mul_f32(0.5), queue, depth - 1);
            }

            return;
        }

        // If the duration is still too long, clamp it to 1/60th of a second
        let timestep = if timestep.as_secs_f32() > 0.016 {
            Duration::from_secs_f32(0.016)
        } else {
            timestep
        };

        let mut forces: Vec<cgmath::Vector3<f32>> = vec![[0.0; 3].into(); self.mesh.vertices.len()];

        Self::apply_gravity(&mut forces, &self.mesh.vertices, self.mesh.settings.gravity);
        Self::apply_spring_damper_forces(&self.mesh.vertices, &self.mesh.springs, &mut forces);
        Self::apply_aerodynamic_forces(
            &self.mesh.vertices,
            &self.mesh.triangles,
            &self.mesh.settings,
            &mut forces,
        );

        self.integrate_forces(&forces, timestep);

        self.update_normals(queue);
    }

    fn apply_gravity(
        forces: &mut Vec<cgmath::Vector3<f32>>,
        vertices: &Vec<Vertex>,
        gravity: [f32; 3],
    ) {
        let gravity: cgmath::Vector3<f32> = gravity.into();

        forces
            .par_iter_mut()
            .zip(vertices)
            .for_each(|(force, vertex)| {
                *force += gravity * vertex.mass;
            });
    }

    fn apply_spring_damper_forces(
        vertices: &Vec<Vertex>,
        springs: &Vec<Spring>,
        forces: &mut Vec<cgmath::Vector3<f32>>,
    ) {
        let (tx, rx) = channel::<(u16, u16, cgmath::Vector3<f32>)>();

        springs.par_iter().for_each_with(tx, |tx, spring| {
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
                tx.send((i_1 as u16, i_2 as u16, f_s + f_d)).unwrap();
            }
        });

        while let Ok((i_1, i_2, spring_force)) = rx.recv() {
            forces[i_1 as usize] += spring_force;
            forces[i_2 as usize] -= spring_force;
        }
    }

    fn apply_aerodynamic_forces(
        vertices: &Vec<Vertex>,
        triangles: &Vec<Triangle>,
        settings: &SimulationSettings,
        forces: &mut Vec<cgmath::Vector3<f32>>,
    ) {
        const COEFFICIENT: f32 = 1.28;
        const ONE_THIRD: f32 = 1.0 / 3.0;

        let (tx, rx) = channel::<(u16, u16, u16, cgmath::Vector3<f32>)>();

        triangles.par_iter().for_each_with(tx, |tx, triangle| {
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
            let area = area * v.normalize().dot(n);

            let f_a = -0.5 * density * v.magnitude2() * COEFFICIENT * area * n;

            // Add forces to vertices
            if f_a.magnitude2().is_finite() {
                tx.send((i_1 as u16, i_2 as u16, i_3 as u16, f_a * ONE_THIRD))
                    .unwrap();
            }
        });

        while let Ok((i_1, i_2, i_3, aerodynamic_force_per_vertex)) = rx.recv() {
            forces[i_1 as usize] += aerodynamic_force_per_vertex;
            forces[i_2 as usize] += aerodynamic_force_per_vertex;
            forces[i_3 as usize] += aerodynamic_force_per_vertex;
        }
    }

    fn integrate_forces(&mut self, forces: &Vec<cgmath::Vector3<f32>>, timestep: Duration) {
        // Integrate forces
        self.mesh
            .vertices
            .par_iter_mut()
            .zip(forces)
            .for_each(|(vertex, force)| {
                let acceleration = *force / vertex.mass;

                if vertex.fixed != 0 {
                    vertex.velocity = [0.0; 3];
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

                    Self::integrate_with_ground_collision(
                        &self.mesh.settings,
                        vertex,
                        force,
                        &new_postion,
                        timestep,
                    );
                }
            });
    }

    fn integrate_with_ground_collision(
        settings: &SimulationSettings,
        vertex: &mut Vertex,
        force: &cgmath::Vector3<f32>,
        new_position: &cgmath::Point3<f32>,
        timestep: Duration,
    ) {
        // Height above the rendered ground to collide with to avoid Z-fighting
        const GROUND_LEVEL_EPSILON: f32 = 0.001;
        // Height above collision ground for friction
        const FRICTION_GROUND_LEVEL_EPSILON: f32 = 0.005;
        let ground_level = settings.ground_level + GROUND_LEVEL_EPSILON;
        let friction_ground_level = settings.ground_level + FRICTION_GROUND_LEVEL_EPSILON;

        // Threshold of movement before friction is applied
        const STATIC_FRICTION_THRESHOLD: f32 = 0.01;

        if new_position[1] < friction_ground_level {
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

            if new_position[1] < ground_level {
                // collision
                let penetration = ground_level - new_position[1];
                vertex.position[1] = ground_level + settings.ground_restitution * penetration;
                vertex.velocity[1] = -vertex.velocity[1] * settings.ground_restitution;

                vertex.position[0] = new_position[0];
                //vertex.position[1] = new_position[1];
                vertex.position[2] = new_position[2];
            } else {
                vertex.position[0] = new_position[0];
                vertex.position[1] = new_position[1];
                vertex.position[2] = new_position[2];
            }
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
