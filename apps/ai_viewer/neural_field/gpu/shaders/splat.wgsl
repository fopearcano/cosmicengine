// Gaussian splat WGSL shader (Phase 27).
//
// Each Gaussian point is rendered as a 6-vertex screen-aligned quad
// (two triangles). The vertex stage computes the warped direction +
// projected position, sizes the quad to a 3-sigma footprint in pixels,
// and emits the corner offset (-1..1) as a varying. The fragment
// stage uses that offset to evaluate the Gaussian kernel and writes
// pre-multiplied additive color.
//
// Storage layout (matches FLOATS_PER_POINT in webgpu_buffer.py):
//   pos_intensity.xyz = world position
//   pos_intensity.w   = intensity
//   color_sigma.xyz   = color in [0, 255]
//   color_sigma.w     = world-space sigma
//
// Uniforms:
//   cam_pos   .w = fov_scale (= tan(fov/2))
//   forward   .w = aspect (= height / width)
//   up        .w = render width (px)
//   right     .w = render height (px)
//   params: x = beta, y = warp_factor,
//           z = enable_warp (0 or 1), w = enable_doppler (0 or 1)

struct Uniforms {
    cam_pos: vec4<f32>,
    forward: vec4<f32>,
    up: vec4<f32>,
    right: vec4<f32>,
    params: vec4<f32>,
};

struct GaussianPoint {
    pos_intensity: vec4<f32>,
    color_sigma: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> points: array<GaussianPoint>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) corner: vec2<f32>,
    @location(2) intensity: f32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VertexOut {
    var out: VertexOut;
    let p = points[iid];
    let world_pos = p.pos_intensity.xyz;
    let intensity = p.pos_intensity.w;
    var color = p.color_sigma.xyz;
    let sigma = p.color_sigma.w;

    let fov_scale = uniforms.cam_pos.w;
    let aspect = uniforms.forward.w;
    let width = uniforms.up.w;
    let height = uniforms.right.w;
    let beta = uniforms.params.x;
    let warp_factor = uniforms.params.y;
    let enable_warp = uniforms.params.z;
    let enable_doppler = uniforms.params.w;

    let cam_pos = uniforms.cam_pos.xyz;
    let forward = uniforms.forward.xyz;
    let up = uniforms.up.xyz;
    let right = uniforms.right.xyz;

    let world_d = world_pos - cam_pos;
    let dist = length(world_d);
    var direction = forward;
    if (dist > 0.0) {
        direction = world_d / dist;
    }

    if (enable_warp > 0.5) {
        let alignment = dot(direction, forward);
        let amount = beta * alignment * warp_factor;
        direction = normalize(direction + forward * amount);
        if (enable_doppler > 0.5) {
            let shift = clamp(beta * alignment * warp_factor, -1.0, 1.0);
            if (shift > 0.0) {
                color.b = color.b + (255.0 - color.b) * shift;
                color.r = color.r * (1.0 - shift);
            } else if (shift < 0.0) {
                let s = -shift;
                color.r = color.r + (255.0 - color.r) * s;
                color.b = color.b * (1.0 - s);
            }
        }
    }

    let warped_world = cam_pos + direction * dist;
    let d = warped_world - cam_pos;
    let f = dot(d, forward);

    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0)
    );
    let corner = corners[vid];

    if (f <= 0.0) {
        // Behind camera; emit a degenerate clip-space vertex.
        out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.color = vec3<f32>(0.0);
        out.corner = vec2<f32>(0.0);
        out.intensity = 0.0;
        return out;
    }

    let r_proj = dot(d, right);
    let u_proj = dot(d, up);
    let nx = (r_proj / f) / fov_scale;
    let ny = (u_proj / f) / (fov_scale * aspect);

    let screen_sigma = max(1.0, sigma * (width / (2.0 * fov_scale)) / f);
    let half_quad_x = (screen_sigma * 3.0) * 2.0 / width;
    let half_quad_y = (screen_sigma * 3.0) * 2.0 / height;

    out.position = vec4<f32>(
        nx + corner.x * half_quad_x,
        ny + corner.y * half_quad_y,
        0.0,
        1.0
    );
    out.color = color / 255.0;
    out.corner = corner;
    out.intensity = intensity;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // corner ∈ [-1, 1] spans the 3-sigma quad footprint. Convert to a
    // distance in sigma units and evaluate the Gaussian kernel.
    let r2 = dot(in.corner, in.corner) * 9.0;
    let kernel = exp(-r2 * 0.5);
    let alpha = clamp(in.intensity * kernel, 0.0, 1.0e6);
    return vec4<f32>(in.color * alpha, alpha);
}
