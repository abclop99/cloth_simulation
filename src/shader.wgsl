struct CameraUniform {
	view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
	@location(0) position: vec3<f32>,
	@location(1) color: vec3<f32>,
	@location(2) normal: vec3<f32>,
	@location(3) mass: f32,
};

struct VertexOutput {
	@builtin(position) clip_position: vec4<f32>,
	@location(0) color: vec3<f32>,
};

@vertex
fn vert_main(
	model: VertexInput,
) -> VertexOutput {
	var out: VertexOutput;
	out.color = model.color +  0.25 * model.normal;
	out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    return out;
}

@fragment
fn frag_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
