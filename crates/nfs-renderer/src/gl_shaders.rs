//! ReNFS5: the GLSL ES 3.00 shaders of the native GLES renderer (`gl_core.rs`).
//!
//! The mesh shader is the port's `shader.wgsl` written for GLES, split into variants: a feature an
//! item does not use (the headlight and beacon projectors, the light layer, the specular, sky-map
//! and shadow passes, the headlight decal, fog, the alpha test) is not compiled into its program at
//! all. Mali-class GPUs pay for every instruction of a shader whether a branch is taken or not.
//! The maths of each feature is the WGSL's, reference by reference; the one deliberate change is
//! that the sun's diffuse term is made per vertex (gouraud, as the original's Direct3D 7 did) for
//! the few items without the light layer — the cars and the track take the layer's per-vertex
//! colour anyway.

/// Feature bits of a mesh program.
pub(crate) mod f {
    /// output: 0 opaque (alpha 1), 1 blended (alpha), 2 the layered passes (premultiplied)
    pub const OUT_MASK: u32 = 3;
    pub const ALPHA_TEST: u32 = 1 << 2;
    pub const HEADLIGHT: u32 = 1 << 3;
    pub const BEACON: u32 = 1 << 4;
    pub const UVLIT: u32 = 1 << 5;
    pub const SHADOW: u32 = 1 << 6;
    pub const SKY: u32 = 1 << 7;
    pub const ENV: u32 = 1 << 8;
    pub const LAYERLIGHT: u32 = 1 << 9;
    pub const FOG: u32 = 1 << 10;
}

pub(crate) fn mesh_sources(key: u32) -> (String, String) {
    let mut defs = String::new();
    let mut def = |name: &str, on: bool| {
        if on {
            defs.push_str("#define ");
            defs.push_str(name);
            defs.push('\n');
        }
    };
    def("OUT_OPAQUE", key & f::OUT_MASK == 0);
    def("OUT_MAIN", key & f::OUT_MASK == 1);
    def("OUT_LAYERS", key & f::OUT_MASK == 2);
    def("ALPHA_TEST", key & f::ALPHA_TEST != 0);
    def("HEADLIGHT", key & f::HEADLIGHT != 0);
    def("BEACON", key & f::BEACON != 0);
    def("UVLIT", key & f::UVLIT != 0);
    def("SHADOW", key & f::SHADOW != 0);
    def("SKY", key & f::SKY != 0);
    def("ENV", key & f::ENV != 0);
    def("LAYERLIGHT", key & f::LAYERLIGHT != 0);
    def("FOG", key & f::FOG != 0);
    (format!("#version 300 es\n{defs}{MESH_VS}"), format!("#version 300 es\n{defs}{MESH_FS}"))
}

const MESH_VS: &str = r#"
precision highp float;
precision highp int;
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in vec4 a_color;

uniform mat4 u_mvp;
uniform mat4 u_model;
uniform mat4 u_view;
// xyz the sun's direction, w the scene's ambient
uniform vec4 u_sun;
// rgb the item's light, a its alpha
uniform vec4 u_light;
// xy the uv offset, z the LOD bias, w the alpha test's reference + 1/2 (0: none)
uniform vec4 u_misc;
// x the depth bias's pull (screen z), y the item's own fog W (0: the view depth), z the y flip
uniform vec4 u_misc2;

out vec2 v_uv;
out vec4 v_color;
out vec3 v_light;
out float v_depth;

#if defined(HEADLIGHT) || defined(BEACON) || defined(UVLIT)
// `night.psh`'s `nght`, 128 x 128 bytes
uniform highp sampler2D t_night;
float night_byte(int row, int col) {
    return floor(texelFetch(t_night, ivec2(col, row), 0).r * 255.0 + 0.5);
}
#endif
#if defined(HEADLIGHT) || defined(UVLIT)
uniform mat4 u_headlight;
// 0x452910
float headlight_w(vec3 world) {
    vec4 p = u_headlight * vec4(world, 1.0);
    int row = int(round(p.z * 1.28));
    int col = int(round(p.x * 3.62));
    if (row <= 0 || row >= 127 || col <= -63 || col >= 63) {
        return 0.0;
    }
    float y = abs(round(p.y * 32.0));
    return max((night_byte(row, col + 64) - y) * 0.00390625, 0.0);
}
#endif
#ifdef HEADLIGHT
vec4 headlit(vec3 world, vec4 c) {
    float w = headlight_w(world);
    if (w <= 0.0) {
        return vec4(c.rgb, 1.0);
    }
    float add = round(128.0 * w);
    return vec4(min(round(c.rgb * 255.0) + vec3(add), vec3(255.0)) / 255.0, 1.0);
}
#endif
#ifdef BEACON
uniform mat4 u_beacon;
// 0x452A50
vec4 beaconlit(vec3 world, vec4 c) {
    vec4 p = u_beacon * vec4(world, 1.0);
    int row = int(round(p.z * 3.5));
    int col = int(round(p.x * 3.5));
    if (row <= -127 || row >= 127 || col <= -63 || col >= 63) {
        return vec4(c.rgb, 1.0);
    }
    float w = (night_byte(abs(row), col + 64) - round(p.y * 16.0)) * 0.001953125;
    if (w <= 0.0) {
        return vec4(c.rgb, 1.0);
    }
    vec3 tint = p.z < 0.0 ? vec3(0.0, 0.0, 255.0) : vec3(255.0, 0.0, 0.0);
    return vec4(min(round(c.rgb * 255.0) + round(tint * w), vec3(255.0)) / 255.0, 1.0);
}
#endif
#ifdef UVLIT
out vec3 v_head_uvw;
out float v_head_w;
// `Headlights::LayerUVLight` 0x453CA0: (u·w, v·w, w), divided per fragment
vec3 uvlight_vertex(vec3 world) {
    vec4 p = u_headlight * vec4(world, 1.0);
    float w = p.z + 16.0;
    float i = 1.0 / w;
    if (i > 0.1) {
        i = 0.1;
        w = 10.0;
    }
    vec2 uv = vec2(p.x * 0.1 + 0.5, 0.0);
    if (i >= 0.0) {
        uv = vec2(i * p.x + 0.5, i * p.y + 0.5);
    } else {
        w = 10.0;
    }
    return vec3(uv * w, w);
}
#endif
#ifdef LAYERLIGHT
uniform vec4 u_lit_a;
uniform vec4 u_lit_d;
uniform vec4 u_lit_dir;
// `R3DLightLayer` 0x446530
vec3 layer_light(vec3 n) {
    float d = 0.0;
    if (length(u_lit_dir.xyz) > 0.0) {
        d = max(dot(n, -normalize(u_lit_dir.xyz)), 0.0);
    }
    return min(floor(u_lit_a.rgb * 255.0 + u_lit_d.rgb * 255.0 * d + vec3(0.5)), vec3(255.0)) / 255.0;
}
#endif
#ifdef SKY
uniform vec4 u_sky_u;
uniform vec4 u_sky_v;
out vec2 v_sky;
#endif
#ifdef ENV
out vec2 v_env;
#endif
#ifdef SHADOW
out vec3 v_world;
#endif

void main() {
    vec4 world = u_model * vec4(a_pos, 1.0);
    vec4 clip = u_mvp * vec4(a_pos, 1.0);
    // Thrash state 0x18 as an offset of the screen z (`shader.wgsl`)
    float dz = u_misc2.x;
    if (dz != 0.0 && clip.w > 0.0 && clip.z >= 0.0 && clip.z <= clip.w) {
        clip.z = clamp(clip.z + dz * clip.w, 0.0, clip.w);
    }
    // the port's projection is Direct3D's (z 0…1); GL clips z on -w…w
    clip.z = 2.0 * clip.z - clip.w;
    clip.y *= u_misc2.z;
    gl_Position = clip;
    vec3 n = normalize((u_model * vec4(a_normal, 0.0)).xyz);
    v_uv = a_uv + u_misc.xy;
    vec4 col = a_color;
#ifdef HEADLIGHT
    col = headlit(world.xyz, col);
#endif
#ifdef BEACON
    col = beaconlit(world.xyz, col);
#endif
    v_color = col;
#ifdef LAYERLIGHT
    v_light = layer_light(n);
#else
    float diffuse = max(dot(n, normalize(u_sun.xyz)), 0.0);
    v_light = u_sun.w * u_light.rgb + vec3(1.0 - u_sun.w) * diffuse;
#endif
    v_depth = u_misc2.y > 0.0 ? u_misc2.y : max(-(u_view * world).z, 0.0);
#ifdef SKY
    vec3 p = (u_model * vec4(a_normal + 3.33 * a_pos, 1.0)).xyz;
    v_sky = vec2(dot(u_sky_u.xyz, p) + u_sky_u.w, dot(u_sky_v.xyz, p) + u_sky_v.w);
#endif
#ifdef ENV
    vec3 vne = (u_view * vec4(n, 0.0)).xyz;
    float vr = inversesqrt(abs(vne.z) + 1.0) * 0.5;
    v_env = vec2(vne.x * vr + 0.5, vne.y * vr + 0.5);
#endif
#ifdef UVLIT
    v_head_w = headlight_w(world.xyz);
    v_head_uvw = uvlight_vertex(world.xyz);
#endif
#ifdef SHADOW
    v_world = world.xyz;
#endif
}
"#;

const MESH_FS: &str = r#"
precision mediump float;
precision highp int;
uniform sampler2D t_diffuse;
uniform vec4 u_light;
uniform vec4 u_misc;
// rgb the fog's colour, w its density
uniform vec4 u_fog;
in vec2 v_uv;
in vec4 v_color;
in vec3 v_light;
in highp float v_depth;
out vec4 o_color;
#ifdef ENV
uniform sampler2D t_env;
in vec2 v_env;
#endif
#ifdef SKY
uniform sampler2D t_sky;
in vec2 v_sky;
#endif
#ifdef UVLIT
uniform sampler2D t_head;
in highp vec3 v_head_uvw;
in float v_head_w;
#endif
#ifdef SHADOW
uniform sampler2D t_proj;
uniform highp vec4 u_shadow_u;
uniform highp vec4 u_shadow_v;
in highp vec3 v_world;
#endif

void main() {
    vec4 tex = texture(t_diffuse, v_uv, u_misc.z);
    vec3 light = v_light;
    vec4 base = tex * v_color;
    vec4 env_over = vec4(0.0);
#ifdef ENV
    vec4 env = textureLod(t_env, v_env, 0.0);
    if (env.a * 255.0 > 10.0) {
        env_over = env;
    }
#endif
    vec3 lit = base.rgb * light;
    float base_a = base.a * u_light.a;
#ifdef OUT_LAYERS
    float cov = base_a;
    if (u_misc.w > 0.0 && base_a * 255.0 < u_misc.w) {
        cov = 0.0;
    }
    lit = lit * cov;
    lit = lit * (1.0 - env_over.a) + env_over.rgb * light * env_over.a;
    cov = cov * (1.0 - env_over.a) + env_over.a;
#else
    lit = mix(lit, env_over.rgb * light, env_over.a);
#endif
#ifdef SKY
    vec4 sky = textureLod(t_sky, v_sky, 0.0);
    if (sky.a * 255.0 > 10.0) {
#ifdef OUT_LAYERS
        lit = lit * (1.0 - sky.a) + sky.rgb * light * sky.a;
        cov = cov * (1.0 - sky.a) + sky.a;
#else
        lit = mix(lit, sky.rgb * light, sky.a);
#endif
    }
#endif
#ifdef UVLIT
    if (v_head_w > 0.0) {
        lit = lit * 2.0 * textureLod(t_head, v_head_uvw.xy / v_head_uvw.z, 0.0).rgb;
    }
#endif
#ifdef SHADOW
    vec2 suv = vec2(dot(u_shadow_u.xyz, v_world) + u_shadow_u.w, dot(u_shadow_v.xyz, v_world) + u_shadow_v.w);
    vec4 sh = texture(t_proj, suv, -1.0);
    if (sh.a * 255.0 > 10.0) {
#ifdef OUT_LAYERS
        lit = lit * (1.0 - sh.a) + sh.rgb * light * sh.a;
        cov = cov * (1.0 - sh.a) + sh.a;
#else
        lit = mix(lit, sh.rgb * light, sh.a);
#endif
    }
#endif
#ifdef OUT_LAYERS
    vec4 c = vec4(lit, cov);
#ifdef FOG
    float fl = exp(-u_fog.w * v_depth);
    c = vec4(c.rgb * fl + u_fog.rgb * (1.0 - fl) * c.a, c.a);
#endif
    o_color = c;
#else
    vec4 c = vec4(lit, base_a);
#ifdef ALPHA_TEST
    if (c.a * 255.0 < u_misc.w) {
        discard;
    }
#endif
#ifdef FOG
    c.rgb = mix(u_fog.rgb, c.rgb, exp(-u_fog.w * v_depth));
#endif
#ifdef OUT_OPAQUE
    o_color = vec4(c.rgb, 1.0);
#else
    o_color = c;
#endif
#endif
}
"#;

pub(crate) const SPRITE_VS: &str = r#"#version 300 es
precision highp float;
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_color;
layout(location = 3) in float a_ref;
out vec2 v_uv;
out vec4 v_color;
flat out float v_ref;
void main() {
    // clip space with a Direct3D z (0…1)
    gl_Position = vec4(a_pos.xy, 2.0 * a_pos.z - 1.0, 1.0);
    v_uv = a_uv;
    v_color = a_color;
    v_ref = a_ref;
}
"#;

pub(crate) const SPRITE_FS: &str = r#"#version 300 es
precision mediump float;
uniform sampler2D t_sprite;
in vec2 v_uv;
in vec4 v_color;
flat in float v_ref;
out vec4 o_color;
void main() {
    vec4 c = texture(t_sprite, v_uv) * v_color;
#ifdef ALPHA_TEST
    if (round(clamp(c.a, 0.0, 1.0) * 255.0) <= v_ref) {
        discard;
    }
#endif
    o_color = c;
}
"#;

/// The display ramp of state 0x2E (`gpu.rs`'s `GammaPass`).
pub(crate) const GAMMA_VS: &str = r#"#version 300 es
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"#;

pub(crate) const GAMMA_FS: &str = r#"#version 300 es
precision highp float;
uniform sampler2D t_src;
uniform float u_gamma;
out vec4 o_color;
void main() {
    vec4 c = texelFetch(t_src, ivec2(gl_FragCoord.xy), 0);
    vec3 i = round(c.rgb * 255.0);
    vec3 w = min(floor(i * 257.0 * u_gamma), vec3(65535.0));
    o_color = vec4(w / 65535.0, c.a);
}
"#;
