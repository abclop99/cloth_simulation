use instant::Duration;
use wgpu::util::DeviceExt;
use winit::event::*;

pub struct Camera {
    pub eye: cgmath::Point3<f32>,
    pub target: cgmath::Point3<f32>,
    pub up: cgmath::Vector3<f32>,
    pub fovy: f32,
    pub aspect: f32,
    pub znear: f32,
    pub zfar: f32,

    pub camera_uniform: CameraUniform,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    pub camera_bind_group: wgpu::BindGroup,

    pub camera_controller: CameraController,
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

impl Camera {
    pub fn update(&mut self, timestep: Duration, queue: &wgpu::Queue) {
        self.eye = self.camera_controller.update_camera(&self, timestep);

        self.update_view_proj();
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    pub fn new(
        device: &wgpu::Device,
        eye: cgmath::Point3<f32>,
        target: cgmath::Point3<f32>,
        up: cgmath::Vector3<f32>,
        fovy: f32,
        aspect: f32,
        znear: f32,
        zfar: f32,
    ) -> Self {
        let camera_uniform = CameraUniform {
            view_proj: Self::build_view_projection_matrix_fields(
                eye, target, up, fovy, aspect, znear, zfar,
            )
            .into(),
        };

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let camera_controller = CameraController::new(5.0);

        let camera = Self {
            eye,
            target,
            up,
            fovy,
            aspect,
            znear,
            zfar,
            camera_uniform,
            camera_buffer,
            camera_bind_group_layout,
            camera_bind_group,
            camera_controller,
        };

        camera
    }

    pub fn update_view_proj(&mut self) {
        self.camera_uniform.view_proj = self.build_view_projection_matrix().into();
    }

    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        Self::build_view_projection_matrix_fields(
            self.eye,
            self.target,
            self.up,
            self.fovy,
            self.aspect,
            self.znear,
            self.zfar,
        )
    }
    fn build_view_projection_matrix_fields(
        eye: cgmath::Point3<f32>,
        target: cgmath::Point3<f32>,
        up: cgmath::Vector3<f32>,
        fovy: f32,
        aspect: f32,
        znear: f32,
        zfar: f32,
    ) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(eye, target, up);
        let proj = cgmath::perspective(cgmath::Deg(fovy), aspect, znear, zfar);
        return OPENGL_TO_WGPU_MATRIX * proj * view;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

pub struct CameraController {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}

impl CameraController {
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }

    pub fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    VirtualKeyCode::W | VirtualKeyCode::Up => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::A | VirtualKeyCode::Left => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::S | VirtualKeyCode::Down => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::D | VirtualKeyCode::Right => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    // Returns new eye
    pub fn update_camera(&self, camera: &Camera, timestep: Duration) -> cgmath::Point3<f32> {
        use cgmath::InnerSpace;
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();
        let timestep = timestep.as_secs_f32();

        let mut eye = camera.eye;

        // Prevents glitching when camera gets too close to the
        // center of the scene.
        if self.is_forward_pressed && forward_mag > (self.speed * timestep) {
            eye += forward_norm * self.speed * timestep;
        }
        if self.is_backward_pressed {
            eye -= forward_norm * self.speed * timestep;
        }

        let right = forward_norm.cross(camera.up);

        // Redo radius calc in case the fowrard/backward is pressed.
        let forward = camera.target - camera.eye;
        let forward_mag = forward.magnitude();

        if self.is_right_pressed {
            // Rescale the distance between the target and eye so
            // that it doesn't change. The eye therefore still
            // lies on the circle made by the target and eye.
            eye =
                camera.target - (forward + right * self.speed * timestep).normalize() * forward_mag;
        }
        if self.is_left_pressed {
            eye =
                camera.target - (forward - right * self.speed * timestep).normalize() * forward_mag;
        }

        eye
    }
}
