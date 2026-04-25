// Geodesic-marched Gaussian splat shader (Phase 29).
//
// Like gr_splat.wgsl but the single weak-field deflection is replaced
// by a fixed-count loop that integrates the photon trajectory toward
// the black hole. The loop length is a compile-time constant
// (GEODESIC_STEPS) so the GPU never has dynamic branching on step
// count; runtime sets `bh_params.w` to the per-step length.

const GEODESIC_STEPS: u32 = 8u;
const MAX_STEP_DEFLECTION: f32 = 0.7853981;  // pi / 4

struct Uniforms {
    cam_pos: vec4<f32>,        // xyz pos, w = fov_scale
    forward: vec4<f32>,        // xyz forward, w = aspect
    up: vec4<f32>,             // xyz up, w = width
    right: vec4<f32>,          // xyz right, w = height
    params: vec4<f32>,         // beta, warp, enable_warp, enable_doppler
    bh_position: vec4<f32>,    // xyz, w = mass_kg
    bh_params: vec4<f32>,      // x = enable_geodesic, y = R_s, z = 4G/c^2, w = step_size
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
    @location(3) absorbed: f32,
};

fn march_geodesic(
    origin: vec3<f32>,
    initial_direction: vec3<f32>,
    bh_pos: vec3<f32>,
    bh_mass: f32,
    r_s: f32,
    step_size: f32,
    deflection_scale: f32,
) -> vec4<f32> {
    var pos = origin - bh_pos;
    var d = normalize(initial_direction);
    var absorbed: f32 = 0.0;
    for (var i: u32 = 0u; i < GEODESIC_STEPS; i = i + 1u) {
        let r = length(pos);
        if (r <= r_s) {
            absorbed = 1.0;
            break;
        }
        let r_hat = -pos / max(r, 1.0e-9);
        let along = dot(r_hat, d);
        let perp_vec = r_hat - along * d;
        let perp_norm = length(perp_vec);
        if (perp_norm > 0.0) {
            let n_perp = perp_vec / perp_norm;
            // The deflection_scale uniform == 4G/c^2 (Phase 28).
            // Per-step rotation: dtheta = (deflection_scale * M / 2) * step / r^2.
            // The factor of /2 turns the lensing constant 4G/c^2 into
            // the geodesic 2G/c^2 used by integrate_geodesic_step.
            var dtheta = 0.5 * deflection_scale * bh_mass * step_size / (r * r);
            dtheta = clamp(dtheta, 0.0, MAX_STEP_DEFLECTION);
            d = normalize(cos(dtheta) * d + sin(dtheta) * n_perp);
        }
        pos = pos + d * step_size;
    }
    return vec4<f32>(d, absorbed);
}

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

    let bh_pos = uniforms.bh_position.xyz;
    let bh_mass = uniforms.bh_position.w;
    let enable_geodesic = uniforms.bh_params.x;
    let r_s = uniforms.bh_params.y;
    let four_GM_over_c2 = uniforms.bh_params.z;
    let step_size = uniforms.bh_params.w;

    let world_d = world_pos - cam_pos;
    let dist = length(world_d);
    var direction = forward;
    if (dist > 0.0) {
        direction = world_d / dist;
    }
    var absorbed: f32 = 0.0;

    if (enable_geodesic > 0.5) {
        let traced = march_geodesic(
            cam_pos, direction, bh_pos, bh_mass, r_s,
            step_size, four_GM_over_c2,
        );
        direction = traced.xyz;
        absorbed = traced.w;
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
        out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.color = vec3<f32>(0.0);
        out.corner = vec2<f32>(0.0);
        out.intensity = 0.0;
        out.absorbed = absorbed;
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
    out.absorbed = absorbed;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    if (in.absorbed > 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let r2 = dot(in.corner, in.corner) * 9.0;
    let kernel = exp(-r2 * 0.5);
    let alpha = clamp(in.intensity * kernel, 0.0, 1.0e6);
    return vec4<f32>(in.color * alpha, alpha);
}
