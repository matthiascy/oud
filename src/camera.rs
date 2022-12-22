use winit::event::{VirtualKeyCode, WindowEvent};

use crate::state::Response;

pub struct Camera {
    pub eye: glam::Vec3,
    pub target: glam::Vec3,
    pub up: glam::Vec3,
    pub aspect: f32,
    pub vfov: f32,
    pub near: f32,
    pub far: f32,
}

pub const OPENGL_TO_WGPU_MATRIX: glam::Mat4 = glam::Mat4::from_cols_array(&[
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0,
]);

impl Camera {
    pub fn view_mat(&self) -> glam::Mat4 {
        glam::Mat4::look_at_rh(self.eye, self.target, self.up)
    }

    pub fn proj_mat(&self) -> glam::Mat4 {
        glam::Mat4::perspective_rh_gl(self.vfov, self.aspect, self.near, self.far)
    }

    pub fn view_proj_mat(&self) -> glam::Mat4 {
        OPENGL_TO_WGPU_MATRIX * self.proj_mat() * self.view_mat()
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [f32; 16],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: glam::Mat4::IDENTITY.to_cols_array(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.view_proj_mat().to_cols_array();
    }
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

    pub fn process_events(&mut self, event: &WindowEvent) -> Response {
        match event {
            WindowEvent::KeyboardInput { input, .. } => {
                if let Some(keycode) = input.virtual_keycode {
                    let is_pressed = input.state == winit::event::ElementState::Pressed;
                    match keycode {
                        VirtualKeyCode::W | VirtualKeyCode::Up => {
                            self.is_forward_pressed = is_pressed;
                            Response::Handled
                        }
                        VirtualKeyCode::S | VirtualKeyCode::Down => {
                            self.is_backward_pressed = is_pressed;
                            Response::Handled
                        }
                        VirtualKeyCode::A | VirtualKeyCode::Left => {
                            self.is_left_pressed = is_pressed;
                            Response::Handled
                        }
                        VirtualKeyCode::D | VirtualKeyCode::Right => {
                            self.is_right_pressed = is_pressed;
                            Response::Handled
                        }
                        _ => Response::Ignored,
                    }
                } else {
                    Response::Ignored
                }
            }
            _ => Response::Ignored,
        }
    }

    pub fn update_camera(&self, camera: &mut Camera) {
        let forward = camera.target - camera.eye;
        let norm = forward.length();
        let normalised = forward / norm;

        // Avoid glitching when camera gets too close to the center of the scene.
        if self.is_forward_pressed && norm > self.speed {
            camera.eye += normalised * self.speed;
        }

        if self.is_backward_pressed {
            camera.eye -= normalised * self.speed;
        }

        let right = normalised.cross(camera.up).normalize();

        // Recalcuate the forward vector in case the camera moved.
        let forward = camera.target - camera.eye;
        let norm = forward.length();

        // Rescale the distance between the camera and the target so that the camera
        // still lies on the circle made by the target and the eye.
        if self.is_right_pressed {
            camera.eye = camera.target - (forward + right * self.speed).normalize() * norm;
        }

        if self.is_left_pressed {
            camera.eye = camera.target - (forward - right * self.speed).normalize() * norm;
        }
    }
}
