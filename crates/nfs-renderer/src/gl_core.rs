//! ReNFS5: the native OpenGL ES 3 renderer core — the same contract as the wgpu core (`gpu.rs`),
//! drawing the game's scene lists with direct GL calls on the window's SDL2 context.
//!
//! Built for handheld GPUs (Mali-G31 / libmali, where a draw with state churn costs tens of µs of
//! driver time — measured in the ReNFS5 recomp's renderer):
//! * every draw is one `glDrawElements`: mesh indices are rebased onto their arena at upload, and
//!   each vertex buffer carries its own vertex array object, so a mesh switch is one bind;
//! * GL state, textures, samplers, programs and per-draw uniforms are cached, only changes reach
//!   the driver; the per-pass globals go to a program the first time it draws in a pass;
//! * mesh programs are compiled per feature set (`gl_shaders.rs`): an item pays only for the
//!   features it uses;
//! * per-frame data (the dynamic meshes' vertices, the sprite stream) is re-specified with
//!   `glBufferData` (a fresh allocation the driver hands out, never a copy of a buffer in use).
//!
//! The draw order, the stages (sky, opaque, blended, overlay), the blend / depth / cull states, the
//! fake-sort double draw, the mirror view's rules, the lens-flare read-backs, the gamma ramp and the
//! offscreen passes are the wgpu core's, read line by line.

use crate::front::{Cmd, CmdList};
use crate::gl_shaders::{self, f};
use crate::gpu::{MaterialId, MeshId, OffscreenSpec, TextureId, VIEW_FAR, still_size};
use crate::{Camera, FakeSortMode, LensProbe, LensRead, MaterialDesc, MeshData, Scene, Sprite, SpriteCompare, Vertex, depth_bias_offset};
use anyhow::{Context, Result, anyhow};
use glow::HasContext;
use std::collections::HashMap;
use std::sync::Arc;
use winit::window::Window;

const VERTEX_STRIDE: u64 = std::mem::size_of::<Vertex>() as u64;
const ARENA_CHUNK: u64 = 4 << 20;
const MAX_OBJECTS: usize = 8192;
const MAX_SPRITES: usize = 8192;

/// Texture units: the mesh program's samplers, and a scratch unit for uploads.
const U_DIFFUSE: u32 = 0;
const U_ENV: u32 = 1;
const U_SKY: u32 = 2;
const U_PROJ: u32 = 3;
const U_HEAD: u32 = 4;
const U_NIGHT: u32 = 5;
const U_SCRATCH: u32 = 7;
const UNITS: usize = 8;

struct GlMesh {
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    /// byte offset of the first index in the element buffer (indices already rebased)
    index_offset: u64,
    index_count: u32,
    vertex_offset: u64,
    vertex_len: u64,
    /// the size the dynamic mesh's buffer was made with (re-specified at that size per write)
    dynamic_size: Option<u64>,
    writes: u32,
    /// ReNFS5: an arena mesh's vertices once it is rewritten — it then has buffers of its own,
    /// re-specified whole per write (`write_mesh`)
    shadow: Option<Vec<u8>>,
    /// ReNFS5: the stream third its VAO reads from, when it reads from the stream
    stream_seg: Option<usize>,
    /// ReNFS5: an arena mesh's own (not rebased) indices, kept for `promote_mesh`
    local_indices: Option<Box<[u32]>>,
}

struct Arena {
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    ibo: glow::Buffer,
    vused: u64,
    iused: u64,
    vcap: u64,
    icap: u64,
}

struct GlTex {
    tex: glow::Texture,
    width: u32,
    height: u32,
}

#[derive(Clone, Copy)]
struct GlMat {
    texture: TextureId,
    env: TextureId,
    /// `clamp_u | clamp_v << 1`
    sampler: usize,
    alpha_blend: bool,
    blend_pair: u8,
    device_states: bool,
    lod_bias: f32,
    cull_back: bool,
    view_cull: bool,
    depth_bias: i32,
    false_sort: bool,
    depth_test: bool,
    env_mix: f32,
    env_variant: usize,
    sky_map: bool,
}

/// One cached uniform: its location and the last value sent.
struct Uni<const N: usize> {
    loc: Option<glow::UniformLocation>,
    last: [f32; N],
}

impl<const N: usize> Uni<N> {
    fn new(gl: &glow::Context, p: glow::Program, name: &str) -> Self {
        // SAFETY: a linked program of this context
        Self { loc: unsafe { gl.get_uniform_location(p, name) }, last: [f32::NAN; N] }
    }
    fn reset(&mut self) {
        self.last = [f32::NAN; N];
    }
}

fn set4(gl: &glow::Context, u: &mut Uni<4>, v: [f32; 4]) {
    if u.loc.is_some() && u.last != v {
        // SAFETY: the program owning `u` is current
        unsafe { gl.uniform_4_f32_slice(u.loc.as_ref(), &v) };
        u.last = v;
    }
}

fn set16(gl: &glow::Context, u: &mut Uni<16>, v: [f32; 16]) {
    if u.loc.is_some() && u.last != v {
        // SAFETY: the program owning `u` is current
        unsafe { gl.uniform_matrix_4_f32_slice(u.loc.as_ref(), false, &v) };
        u.last = v;
    }
}

struct MeshProg {
    prog: glow::Program,
    mvp: Uni<16>,
    model: Uni<16>,
    view: Uni<16>,
    headlight: Uni<16>,
    beacon: Uni<16>,
    sun: Uni<4>,
    light: Uni<4>,
    misc: Uni<4>,
    misc2: Uni<4>,
    fog: Uni<4>,
    lit_a: Uni<4>,
    lit_d: Uni<4>,
    lit_dir: Uni<4>,
    sky_u: Uni<4>,
    sky_v: Uni<4>,
    shadow_u: Uni<4>,
    shadow_v: Uni<4>,
    /// the pass whose globals it holds
    pass: u64,
}

struct SpriteProg {
    prog: glow::Program,
}

/// What GL has applied, so a draw sends only changes.
#[derive(Clone, Copy, PartialEq)]
struct Raster {
    /// `None` = blending off; else (src rgb, dst rgb, src a, dst a)
    blend: Option<(u32, u32, u32, u32)>,
    depth_test: bool,
    depth_func: u32,
    depth_write: bool,
    /// 0 none, `glow::BACK`, `glow::FRONT`
    cull: u32,
    front_face: u32,
}

struct GlState {
    program: Option<glow::Program>,
    vao: Option<glow::VertexArray>,
    raster: Option<Raster>,
    textures: [Option<glow::Texture>; UNITS],
    samplers: [Option<glow::Sampler>; UNITS],
    active: u32,
}

impl GlState {
    fn new() -> Self {
        Self { program: None, vao: None, raster: None, textures: [None; UNITS], samplers: [None; UNITS], active: u32::MAX }
    }
}

/// Where a pass draws: a framebuffer (`None` = the window's), its size, and whether its rows are
/// kept top row first (an image the sprite pass later samples with top-left uv).
#[derive(Clone, Copy)]
struct Tgt {
    fbo: Option<glow::Framebuffer>,
    width: u32,
    height: u32,
    flip: bool,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SpriteVertex {
    pos: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
    alpha_ref: f32,
}

struct Offscreen {
    fbo: glow::Framebuffer,
}

pub(crate) struct GlCore {
    window: Arc<Window>,
    gl: glow::Context,
    size: (u32, u32),
    max_tex: u32,
    meshes: Vec<GlMesh>,
    arena: Option<Arena>,
    textures: Vec<GlTex>,
    materials: Vec<GlMat>,
    white: TextureId,
    programs: HashMap<u32, MeshProg>,
    failed_programs: std::collections::HashSet<u32>,
    sprite_progs: [SpriteProg; 2],
    gamma_prog: (glow::Program, Option<glow::UniformLocation>),
    samplers: [glow::Sampler; 8],
    point_samplers: [glow::Sampler; 4],
    proj_samplers: [glow::Sampler; 2],
    env_sampler: glow::Sampler,
    sprite_sampler: glow::Sampler,
    nearest_sampler: glow::Sampler,
    night_tex: glow::Texture,
    proj_default: glow::Texture,
    proj_tex: Option<TextureId>,
    head_tex: Option<TextureId>,
    no_reflections: bool,
    trilinear: bool,
    fakesort_mode: FakeSortMode,
    gamma: f32,
    lens_enabled: bool,
    lens_done: Vec<LensRead>,
    /// ReNFS5: the window frame's lens reads in flight — read into a pixel-pack buffer behind a
    /// fence and collected when the GPU is done, so the CPU never waits on the GPU mid-frame (a
    /// synchronous `glReadPixels` cost 50–70 ms frames on Mali with the sun in view)
    lens_pending: Vec<LensPending>,
    lens_pbos: Vec<glow::Buffer>,
    /// the window frame (`render`) reads asynchronously; stills and images keep the direct read
    lens_async: bool,
    /// ReNFS5: the per-frame vertex stream for rewritten meshes (`write_mesh`)
    stream: Option<StreamRing>,
    /// ReNFS5: every arena chunk's element buffer by its vertex buffer (`promote_mesh` reads any)
    arena_ibos: HashMap<glow::Buffer, glow::Buffer>,
    /// ReNFS5: compiled mesh programs on disk (`NFS5_SHADER_CACHE`)
    shader_cache: Option<ShaderCache>,
    offscreen: HashMap<u32, (u32, u32)>,
    offscreen_fbos: HashMap<u32, Offscreen>,
    depth_rbs: HashMap<(u32, u32), glow::Renderbuffer>,
    /// the frame drawn before the gamma ramp: (w, h, fbo, colour texture)
    gamma_target: Option<(u32, u32, glow::Framebuffer, glow::Texture)>,
    sprite_vao: glow::VertexArray,
    /// no attributes: the gamma pass's full-screen triangle
    empty_vao: glow::VertexArray,
    sprite_vbo: glow::Buffer,
    sprite_scratch: Vec<SpriteVertex>,
    buckets: [Vec<u32>; 4],
    pass: u64,
    st: GlState,
    stats_draws: u64,
    stats_t: std::time::Instant,
}

// SAFETY: the context is used by one thread at a time (the core is handed over whole, with the
// context made current on the receiving side, `front.rs`)
unsafe impl Send for GlCore {}

fn compile(gl: &glow::Context, vs: &str, fs: &str, attribs: &[(u32, &str)]) -> Result<glow::Program> {
    // SAFETY: plain GL object creation on the current context
    unsafe {
        let p = gl.create_program().map_err(|e| anyhow!(e))?;
        let mut shaders = Vec::new();
        for (kind, src) in [(glow::VERTEX_SHADER, vs), (glow::FRAGMENT_SHADER, fs)] {
            let s = gl.create_shader(kind).map_err(|e| anyhow!(e))?;
            gl.shader_source(s, src);
            gl.compile_shader(s);
            if !gl.get_shader_compile_status(s) {
                let log = gl.get_shader_info_log(s);
                return Err(anyhow!("shader compile: {log}\n{src}"));
            }
            gl.attach_shader(p, s);
            shaders.push(s);
        }
        for (i, name) in attribs {
            gl.bind_attrib_location(p, *i, name);
        }
        // ReNFS5: the linked program is saved to the shader cache (`ShaderCache`)
        gl.program_binary_retrievable_hint(p, true);
        gl.link_program(p);
        if !gl.get_program_link_status(p) {
            return Err(anyhow!("program link: {}", gl.get_program_info_log(p)));
        }
        for s in shaders {
            gl.detach_shader(p, s);
            gl.delete_shader(s);
        }
        Ok(p)
    }
}

impl GlCore {
    pub fn new(window: Arc<Window>) -> Result<Self> {
        window.gl_context().context("SDL2 GLES context")?;
        let w = window.clone();
        // SAFETY: the context is current on this thread
        let gl = unsafe { glow::Context::from_loader_function(|s| w.gl_get_proc_address(s)) };
        // SAFETY: queries on the current context
        let (renderer, version, max_tex) = unsafe { (gl.get_parameter_string(glow::RENDERER), gl.get_parameter_string(glow::VERSION), gl.get_parameter_i32(glow::MAX_TEXTURE_SIZE)) };
        tracing::info!(renderer = %renderer, version = %version, "native GLES renderer");
        let _ = crate::ADAPTER_LABEL.set(format!("{renderer} · GLES (native)"));
        let size = window.inner_size();
        // SAFETY: GL object creation on the current context
        let core = unsafe {
            let mag_min = if crate::point_filter() { glow::NEAREST } else { glow::LINEAR };
            let sampler = |wrap_u: u32, wrap_v: u32, mag: u32, min: u32, max_lod: Option<f32>| -> Result<glow::Sampler> {
                let s = gl.create_sampler().map_err(|e| anyhow!(e))?;
                gl.sampler_parameter_i32(s, glow::TEXTURE_WRAP_S, wrap_u as i32);
                gl.sampler_parameter_i32(s, glow::TEXTURE_WRAP_T, wrap_v as i32);
                gl.sampler_parameter_i32(s, glow::TEXTURE_MAG_FILTER, mag as i32);
                gl.sampler_parameter_i32(s, glow::TEXTURE_MIN_FILTER, min as i32);
                if let Some(l) = max_lod {
                    gl.sampler_parameter_f32(s, glow::TEXTURE_MAX_LOD, l);
                }
                Ok(s)
            };
            let wrap = |c: bool| if c { glow::CLAMP_TO_EDGE } else { glow::REPEAT };
            // the scene samplers by `clamp_u | clamp_v << 1 | point_mip << 2` (gpu.rs)
            let min_of = |mip_point: bool| match (mag_min == glow::LINEAR, mip_point) {
                (true, false) => glow::LINEAR_MIPMAP_LINEAR,
                (true, true) => glow::LINEAR_MIPMAP_NEAREST,
                (false, false) => glow::NEAREST_MIPMAP_LINEAR,
                (false, true) => glow::NEAREST_MIPMAP_NEAREST,
            };
            let mut samplers = Vec::new();
            for i in 0..8 {
                samplers.push(sampler(wrap(i & 1 != 0), wrap(i & 2 != 0), mag_min, min_of(i & 4 != 0), None)?);
            }
            let mut point_samplers = Vec::new();
            for i in 0..4 {
                point_samplers.push(sampler(wrap(i & 1 != 0), wrap(i & 2 != 0), glow::NEAREST, glow::NEAREST_MIPMAP_NEAREST, None)?);
            }
            let proj_samplers = [sampler(glow::CLAMP_TO_EDGE, glow::CLAMP_TO_EDGE, mag_min, min_of(false), None)?, sampler(glow::CLAMP_TO_EDGE, glow::CLAMP_TO_EDGE, mag_min, min_of(true), None)?];
            let env_sampler = sampler(glow::REPEAT, glow::REPEAT, mag_min, min_of(false), None)?;
            let sprite_sampler = sampler(glow::CLAMP_TO_EDGE, glow::CLAMP_TO_EDGE, mag_min, if mag_min == glow::LINEAR { glow::LINEAR_MIPMAP_NEAREST } else { glow::NEAREST_MIPMAP_NEAREST }, Some(0.0))?;
            let nearest_sampler = sampler(glow::CLAMP_TO_EDGE, glow::CLAMP_TO_EDGE, glow::NEAREST, glow::NEAREST, None)?;

            let tex_rgba = |w: u32, h: u32, px: &[u8], filter: u32| -> Result<glow::Texture> {
                let t = gl.create_texture().map_err(|e| anyhow!(e))?;
                gl.active_texture(glow::TEXTURE0 + U_SCRATCH);
                gl.bind_texture(glow::TEXTURE_2D, Some(t));
                gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, w as i32, h as i32, 0, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(Some(px)));
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAX_LEVEL, 0);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter as i32);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter as i32);
                Ok(t)
            };
            let proj_default = tex_rgba(1, 1, &[255, 255, 255, 0], glow::LINEAR)?;
            let night_tex = gl.create_texture().map_err(|e| anyhow!(e))?;
            gl.bind_texture(glow::TEXTURE_2D, Some(night_tex));
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::R8 as i32, 128, 128, 0, glow::RED, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(Some(&[0u8; 128 * 128])));
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 4);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAX_LEVEL, 0);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.bind_texture(glow::TEXTURE_2D, None);

            let sprite_attrs = [(0, "a_pos"), (1, "a_uv"), (2, "a_color"), (3, "a_ref")];
            let sprite_plain = compile(&gl, gl_shaders::SPRITE_VS, gl_shaders::SPRITE_FS, &sprite_attrs)?;
            let sprite_test = compile(&gl, gl_shaders::SPRITE_VS, &gl_shaders::SPRITE_FS.replacen("#version 300 es\n", "#version 300 es\n#define ALPHA_TEST\n", 1), &sprite_attrs)?;
            for p in [sprite_plain, sprite_test] {
                gl.use_program(Some(p));
                gl.uniform_1_i32(gl.get_uniform_location(p, "t_sprite").as_ref(), 0);
            }
            let gamma = compile(&gl, gl_shaders::GAMMA_VS, gl_shaders::GAMMA_FS, &[])?;
            gl.use_program(Some(gamma));
            gl.uniform_1_i32(gl.get_uniform_location(gamma, "t_src").as_ref(), 0);
            let gamma_loc = gl.get_uniform_location(gamma, "u_gamma");
            gl.use_program(None);

            let sprite_vbo = gl.create_buffer().map_err(|e| anyhow!(e))?;
            let sprite_vao = gl.create_vertex_array().map_err(|e| anyhow!(e))?;
            gl.bind_vertex_array(Some(sprite_vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(sprite_vbo));
            let stride = std::mem::size_of::<SpriteVertex>() as i32;
            for (i, n, off) in [(0u32, 3, 0), (1, 2, 12), (2, 4, 20), (3, 1, 36)] {
                gl.enable_vertex_attrib_array(i);
                gl.vertex_attrib_pointer_f32(i, n, glow::FLOAT, false, stride, off);
            }
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            let empty_vao = gl.create_vertex_array().map_err(|e| anyhow!(e))?;

            GlCore {
                window,
                size: (size.width.max(1), size.height.max(1)),
                max_tex: max_tex.max(1) as u32,
                meshes: Vec::new(),
                arena: None,
                textures: Vec::new(),
                materials: Vec::new(),
                white: TextureId(0),
                programs: HashMap::new(),
                failed_programs: Default::default(),
                sprite_progs: [SpriteProg { prog: sprite_plain }, SpriteProg { prog: sprite_test }],
                gamma_prog: (gamma, gamma_loc),
                samplers: samplers.try_into().map_err(|_| anyhow!("samplers"))?,
                point_samplers: point_samplers.try_into().map_err(|_| anyhow!("samplers"))?,
                proj_samplers,
                env_sampler,
                sprite_sampler,
                nearest_sampler,
                night_tex,
                proj_default,
                proj_tex: None,
                head_tex: None,
                no_reflections: false,
                trilinear: false,
                fakesort_mode: FakeSortMode::default(),
                gamma: 1.0,
                lens_enabled: false,
                lens_done: Vec::new(),
                lens_pending: Vec::new(),
                lens_pbos: Vec::new(),
                lens_async: false,
                stream: None,
                arena_ibos: HashMap::new(),
                shader_cache: ShaderCache::open(&renderer, &version),
                offscreen: HashMap::new(),
                offscreen_fbos: HashMap::new(),
                depth_rbs: HashMap::new(),
                gamma_target: None,
                sprite_vao,
                empty_vao,
                sprite_vbo,
                sprite_scratch: Vec::new(),
                buckets: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
                pass: 0,
                st: GlState::new(),
                stats_draws: 0,
                stats_t: std::time::Instant::now(),
                gl,
            }
        };
        let mut core = core;
        core.white = core.upload_texture_mips(1, 1, &[vec![255, 255, 255, 255]]);
        Ok(core)
    }

    // ---- the front's mirror ------------------------------------------------------------------

    pub(crate) fn white(&self) -> TextureId {
        self.white
    }
    pub(crate) fn texture_count(&self) -> u32 {
        self.textures.len() as u32
    }
    pub(crate) fn mesh_infos(&self) -> Vec<(u64, u32)> {
        self.meshes.iter().map(|m| (m.vertex_len, m.writes)).collect()
    }
    pub(crate) fn mesh_vertex_len(&self, id: MeshId) -> u64 {
        self.meshes[id.0 as usize].vertex_len
    }
    pub(crate) fn material_textures(&self) -> Vec<TextureId> {
        self.materials.iter().map(|m| m.texture).collect()
    }
    pub(crate) fn offscreen_sizes(&self) -> HashMap<u32, (u32, u32)> {
        self.offscreen.clone()
    }
    pub fn size(&self) -> (u32, u32) {
        self.size
    }
    pub(crate) fn take_lens(&mut self) -> Vec<LensRead> {
        self.lens_poll(false);
        std::mem::take(&mut self.lens_done)
    }

    /// ReNFS5: move the finished asynchronous lens reads into `lens_done`, in order. `force` (or a
    /// read three frames old) waits for the GPU instead of leaving it for later.
    fn lens_poll(&mut self, force: bool) {
        while let Some(p) = self.lens_pending.first() {
            let wait = force || self.pass.wrapping_sub(p.pass) > 6;
            // SAFETY: a fence this context made
            let st = unsafe { self.gl.client_wait_sync(p.fence, if wait { glow::SYNC_FLUSH_COMMANDS_BIT } else { 0 }, if wait { 100_000_000 } else { 0 }) };
            if st != glow::ALREADY_SIGNALED && st != glow::CONDITION_SATISFIED && !wait {
                break;
            }
            let p = self.lens_pending.remove(0);
            let mut block = [0u8; 64];
            // SAFETY: the buffer the read went into, 64 bytes, mapped for reading and unmapped
            unsafe {
                self.gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(p.pbo));
                let ptr = self.gl.map_buffer_range(glow::PIXEL_PACK_BUFFER, 0, 64, glow::MAP_READ_BIT);
                if !ptr.is_null() {
                    std::ptr::copy_nonoverlapping(ptr, block.as_mut_ptr(), 64);
                    self.gl.unmap_buffer(glow::PIXEL_PACK_BUFFER);
                }
                self.gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
                self.gl.delete_sync(p.fence);
            }
            self.lens_pbos.push(p.pbo);
            let pixels = lens_block_pixels(&block, p.flip);
            self.lens_done.push(LensRead { reference: p.reference, pixels, cost_ms: 0 });
        }
    }
    pub(crate) fn gl_current(&self, current: bool) {
        self.window.gl_make_current(current);
    }
    /// ReNFS5: the next third of the vertex stream, once the GPU is done with it (its fence was
    /// set three frames ago, so this wait is normally none).
    pub(crate) fn frame_begin(&mut self) {
        let Some(r) = self.stream.as_mut() else { return };
        r.seg = (r.seg + 1) % STREAM_SEGS;
        r.cursor = r.seg as u64 * r.seg_size;
        if let Some(f) = r.fences[r.seg].take() {
            // SAFETY: a fence of this context
            unsafe {
                self.gl.client_wait_sync(f, glow::SYNC_FLUSH_COMMANDS_BIT, 1_000_000_000);
                self.gl.delete_sync(f);
            }
        }
        // a mesh not rewritten since it went into this third still reads from it: back into its
        // own buffer before the third is filled with this frame's writes (else it would draw
        // another mesh's vertices — stray triangles)
        let seg = r.seg;
        let mut residents = std::mem::take(&mut r.residents[seg]);
        if env_off("NFS5_REHOME") {
            residents.clear();
        }
        for id in residents {
            let Some(m) = self.meshes.get_mut(id as usize) else { continue };
            if m.stream_seg != Some(seg) {
                continue;
            }
            m.stream_seg = None;
            let (vao, vbo) = (m.vao, m.vbo);
            let Some(shadow) = m.shadow.as_ref() else { continue };
            // SAFETY: the mesh's own buffer, re-specified (no wait), and its VAO pointed back
            unsafe {
                self.gl.bind_vertex_array(None);
                self.st.vao = None;
                self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
                self.gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, shadow, glow::STREAM_DRAW);
                self.gl.bind_buffer(glow::ARRAY_BUFFER, None);
                Self::point_vao(&self.gl, vao, vbo, 0);
            }
        }
    }

    /// TEST GLUE (ReNFS5): `NFS5_FRAME_DUMP=<dir>` writes every `NFS5_FRAME_DUMP_EVERY`th (30) window
    /// frame as `<dir>/f<frame>.ppm`, read back before the swap — the real frame path, for comparing
    /// builds pixel by pixel with `NFS5_FIXED_DT_MS`.
    fn frame_dump(&mut self, w: u32, h: u32) {
        static DIR: std::sync::OnceLock<Option<(String, u64)>> = std::sync::OnceLock::new();
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let Some((dir, every)) = DIR.get_or_init(|| {
            let d = std::env::var("NFS5_FRAME_DUMP").ok()?;
            let e = std::env::var("NFS5_FRAME_DUMP_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(30u64).max(1);
            let _ = std::fs::create_dir_all(&d);
            Some((d, e))
        }) else { return };
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n % every != 0 {
            return;
        }
        let mut px = vec![0u8; (w * h * 4) as usize];
        // SAFETY: a read of the window's framebuffer
        unsafe {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.gl.read_pixels(0, 0, w as i32, h as i32, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::Slice(Some(&mut px)));
        }
        let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
        for y in (0..h).rev() {
            for x in 0..w {
                let o = ((y * w + x) * 4) as usize;
                out.extend_from_slice(&px[o..o + 3]);
            }
        }
        let _ = std::fs::write(format!("{dir}/f{n:05}.ppm"), out);
    }

    /// ReNFS5: the end of a window frame: the stream's current third is fenced.
    fn stream_fence(&mut self) {
        let Some(r) = self.stream.as_mut() else { return };
        // SAFETY: a fence on the current context
        if let Ok(f) = unsafe { self.gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0) } {
            if let Some(old) = r.fences[r.seg].replace(f) {
                unsafe { self.gl.delete_sync(old) };
            }
        }
    }

    pub(crate) fn apply(&mut self, list: &mut CmdList) {
        let CmdList { cmds, bytes, .. } = list;
        for c in cmds.drain(..) {
            let kind: &'static str = match &c {
                Cmd::WriteMesh { .. } => "c_writemesh",
                Cmd::UpdateTextureLevel { .. } => "c_texlevel",
                Cmd::DynamicMesh(_) => "c_dynmesh",
                Cmd::UploadMesh(_) => "c_uploadmesh",
                Cmd::UploadTextures(_) => "c_uploadtex",
                Cmd::Offscreen(_) => "c_offscreen",
                Cmd::CreateMaterial(_) | Cmd::DuplicateMaterial { .. } => "c_material",
                _ => "c_state",
            };
            crate::hitch::mark("c_gap");
            match c {
                Cmd::WriteMesh { mesh, start, len } => self.write_mesh(mesh, &bytes[start..start + len]),
                Cmd::UpdateTextureLevel { texture, level, width, height, start, len } => self.update_texture_level(texture, level, width, height, &bytes[start..start + len]),
                Cmd::HeadlightMap(map) => self.set_headlight_map(&map),
                Cmd::Projector(t) => self.proj_tex = t,
                Cmd::HeadlightTexture(t) => self.head_tex = t,
                Cmd::MirrorView(on) => self.no_reflections = on,
                Cmd::Trilinear(on) => self.trilinear = on,
                Cmd::FakeSort(m) => self.fakesort_mode = m,
                Cmd::Gamma(g) => self.gamma = if g <= 0.0 { 0.0 } else if g < 4.0 { g } else { 4.0 },
                Cmd::Resize(w, h) => {
                    if w > 0 && h > 0 {
                        self.size = (w, h);
                    }
                }
                Cmd::LensProbe(on) => self.lens_enabled = on,
                Cmd::CreateMaterial(d) => {
                    self.create_material(d);
                }
                Cmd::DuplicateMaterial { id, texture } => {
                    self.duplicate_material(id, texture);
                }
                Cmd::DynamicMesh(m) => {
                    self.upload_dynamic_mesh(&m);
                }
                Cmd::UploadMesh(m) => {
                    self.upload_mesh(&m);
                }
                Cmd::UploadTextures(items) => {
                    let refs: Vec<(u32, u32, &[Vec<u8>])> = items.iter().map(|(w, h, m)| (*w, *h, m.as_slice())).collect();
                    self.upload_textures(&refs);
                }
                Cmd::Offscreen(j) => {
                    let id = self.render_offscreen(&j.scene, &j.camera, OffscreenSpec { width: j.width, height: j.height, reuse: j.reuse, clear_colour: j.clear, depth_far: j.depth_far });
                    debug_assert_eq!(id, j.id, "offscreen id allocated on the front differs from the core's");
                }
            }
            crate::hitch::mark(kind);
        }
        bytes.clear();
    }

    // ---- resources ---------------------------------------------------------------------------

    fn make_vao(&self, vbo: glow::Buffer, ibo: glow::Buffer) -> glow::VertexArray {
        // SAFETY: GL objects of this context
        unsafe {
            let vao = self.gl.create_vertex_array().expect("vertex array");
            self.gl.bind_vertex_array(Some(vao));
            self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            let s = VERTEX_STRIDE as i32;
            for (i, n, off) in [(0u32, 3, 0), (1, 3, 12), (2, 2, 24), (3, 4, 32)] {
                self.gl.enable_vertex_attrib_array(i);
                self.gl.vertex_attrib_pointer_f32(i, n, glow::FLOAT, false, s, off);
            }
            self.gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ibo));
            self.gl.bind_vertex_array(None);
            self.gl.bind_buffer(glow::ARRAY_BUFFER, None);
            vao
        }
    }

    pub fn upload_mesh(&mut self, mesh: &MeshData) -> MeshId {
        let vbytes: &[u8] = bytemuck::cast_slice(&mesh.vertices);
        let vlen = vbytes.len() as u64;
        let ilen = (mesh.indices.len() * 4) as u64;
        let (vreserve, ireserve) = (vlen.next_multiple_of(VERTEX_STRIDE).max(VERTEX_STRIDE), ilen.next_multiple_of(4).max(4));
        let fits = self.arena.as_ref().is_some_and(|a| a.vused + vreserve <= a.vcap && a.iused + ireserve <= a.icap);
        // SAFETY: GL buffer work on the current context; the VAO is unbound around the element
        // buffer bind (it belongs to the arena's own VAO)
        unsafe {
            if !fits {
                let (vcap, icap) = (ARENA_CHUNK.max(vreserve), ARENA_CHUNK.max(ireserve));
                let vbo = self.gl.create_buffer().expect("buffer");
                let ibo = self.gl.create_buffer().expect("buffer");
                self.gl.bind_vertex_array(None);
                self.st.vao = None;
                self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
                self.gl.buffer_data_size(glow::ARRAY_BUFFER, vcap as i32, glow::STATIC_DRAW);
                self.gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ibo));
                self.gl.buffer_data_size(glow::ELEMENT_ARRAY_BUFFER, icap as i32, glow::STATIC_DRAW);
                let vao = self.make_vao(vbo, ibo);
                self.arena_ibos.insert(vbo, ibo);
                self.arena = Some(Arena { vao, vbo, ibo, vused: 0, iused: 0, vcap, icap });
            }
            let a = self.arena.as_mut().expect("arena");
            let base = (a.vused / VERTEX_STRIDE) as u32;
            let indices: Vec<u32> = mesh.indices.iter().map(|i| i + base).collect();
            self.gl.bind_vertex_array(None);
            self.st.vao = None;
            if !vbytes.is_empty() {
                self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(a.vbo));
                self.gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, a.vused as i32, vbytes);
            }
            if !indices.is_empty() {
                self.gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(a.ibo));
                self.gl.buffer_sub_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, a.iused as i32, bytemuck::cast_slice(&indices));
            }
            self.gl.bind_buffer(glow::ARRAY_BUFFER, None);
            self.meshes.push(GlMesh { vao: a.vao, vbo: a.vbo, index_offset: a.iused, index_count: mesh.indices.len() as u32, vertex_offset: a.vused, vertex_len: vlen, dynamic_size: None, writes: 0, shadow: None, stream_seg: None, local_indices: Some(mesh.indices.clone().into_boxed_slice()) });
            a.vused += vreserve;
            a.iused += ireserve;
        }
        MeshId(self.meshes.len() as u32 - 1)
    }

    fn upload_dynamic_mesh(&mut self, mesh: &MeshData) -> MeshId {
        let vbytes: &[u8] = bytemuck::cast_slice(&mesh.vertices);
        // the front's mirror counts this size (`create_buffer_init`'s rounding)
        let vlen = (vbytes.len() as u64).next_multiple_of(4).max(4);
        // SAFETY: GL buffer work on the current context
        unsafe {
            let vbo = self.gl.create_buffer().expect("buffer");
            let ibo = self.gl.create_buffer().expect("buffer");
            self.gl.bind_vertex_array(None);
            self.st.vao = None;
            self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            self.gl.buffer_data_size(glow::ARRAY_BUFFER, vlen as i32, glow::STREAM_DRAW);
            if !vbytes.is_empty() {
                self.gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0, vbytes);
            }
            self.gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ibo));
            let ib: &[u8] = bytemuck::cast_slice(&mesh.indices);
            self.gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, if ib.is_empty() { &[0; 4] } else { ib }, glow::STATIC_DRAW);
            let vao = self.make_vao(vbo, ibo);
            let mut shadow = vbytes.to_vec();
            shadow.resize(vlen as usize, 0);
            self.meshes.push(GlMesh { vao, vbo, index_offset: 0, index_count: mesh.indices.len() as u32, vertex_offset: 0, vertex_len: vlen, dynamic_size: Some(vlen), writes: 0, shadow: Some(shadow), stream_seg: None, local_indices: None });
        }
        MeshId(self.meshes.len() as u32 - 1)
    }

    fn write_mesh(&mut self, id: u32, bytes: &[u8]) {
        let Some(m) = self.meshes.get(id as usize) else { return };
        if bytes.is_empty() {
            return;
        }
        // ReNFS5: a write into the shared arena makes the Mali driver wait for (or copy) the whole
        // arena while the GPU still draws from it — 44 writes a frame cost 50 ms frames in the Alps.
        // The mesh moves to buffers of its own on its first write instead.
        if m.dynamic_size.is_none() && m.local_indices.is_some() && !env_off("NFS5_PROMOTE") {
            self.promote_mesh(id);
        }
        let full = bytes.len() as u64 >= self.meshes[id as usize].vertex_len;
        if self.meshes[id as usize].dynamic_size.is_some() && self.meshes[id as usize].shadow.is_none() && !full {
            // a part-write of a promoted mesh with no kept copy yet: its vertices read back once
            self.read_back_shadow(id);
        }
        let m = &mut self.meshes[id as usize];
        m.writes = m.writes.wrapping_add(1);
        match m.shadow.as_mut() {
            Some(shadow) => {
                let n = bytes.len().min(shadow.len());
                shadow[..n].copy_from_slice(&bytes[..n]);
            }
            None if full && m.dynamic_size.is_some() => m.shadow = Some(bytes[..m.vertex_len as usize].to_vec()),
            None => {}
        }
        if self.stream_write(id) {
            return;
        }
        let m = &mut self.meshes[id as usize];
        // SAFETY: GL buffer work on the current context
        unsafe {
            // the stream is full or unusable: the mesh's own buffer, the VAO pointed back at it
            self.gl.bind_vertex_array(None);
            self.st.vao = None;
            if m.shadow.is_some() {
                Self::point_vao(&self.gl, m.vao, m.vbo, 0);
            }
            m.stream_seg = None;
            self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(m.vbo));
            if let Some(shadow) = m.shadow.as_mut() {
                self.gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, shadow, glow::STREAM_DRAW);
            } else if let Some(size) = m.dynamic_size {
                // a fresh allocation for this frame's vertices: the buffer the last frames drew
                // from is left to the GPU
                self.gl.buffer_data_size(glow::ARRAY_BUFFER, size as i32, glow::STREAM_DRAW);
                self.gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0, bytes);
            } else {
                self.gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, m.vertex_offset as i32, bytes);
            }
            self.gl.bind_buffer(glow::ARRAY_BUFFER, None);
        }
    }

    /// ReNFS5: mesh `id`'s kept vertices into this frame's third of the vertex stream — a mapped
    /// write that never waits (the third is fenced off from the GPU's use) — and its VAO pointed
    /// there. `false` when the stream cannot take it.
    fn stream_write(&mut self, id: u32) -> bool {
        if std::env::var("NFS5_STREAM").is_ok_and(|v| v.trim() == "0") {
            return false;
        }
        let len = match self.meshes.get(id as usize).and_then(|m| m.shadow.as_ref()) {
            Some(sh) if !sh.is_empty() => sh.len() as u64,
            _ => return false,
        };
        if self.stream.is_none() {
            // SAFETY: a buffer of this context
            unsafe {
                let Ok(vbo) = self.gl.create_buffer() else { return false };
                self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
                self.gl.buffer_data_size(glow::ARRAY_BUFFER, (STREAM_SEG_SIZE * STREAM_SEGS as u64) as i32, glow::STREAM_DRAW);
                self.gl.bind_buffer(glow::ARRAY_BUFFER, None);
                self.stream = Some(StreamRing { vbo, seg_size: STREAM_SEG_SIZE, seg: 0, cursor: 0, fences: [None, None, None], residents: Default::default() });
            }
        }
        let r = self.stream.as_mut().expect("stream");
        let at = r.cursor.next_multiple_of(64);
        if at + len > (r.seg as u64 + 1) * r.seg_size {
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!(segment_mb = r.seg_size >> 20, "vertex stream full this frame: the rest of the writes take the slow path");
            }
            return false;
        }
        r.cursor = at + len;
        let seg = r.seg;
        r.residents[seg].push(id);
        let vbo = r.vbo;
        self.meshes[id as usize].stream_seg = Some(seg);
        let m = &self.meshes[id as usize];
        let shadow = m.shadow.as_ref().expect("shadow");
        // SAFETY: a range of the stream no draw in flight reads (its third's fence has passed);
        // mapped, filled and unmapped before any draw
        unsafe {
            self.gl.bind_vertex_array(None);
            self.st.vao = None;
            self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            let p = self.gl.map_buffer_range(glow::ARRAY_BUFFER, at as i32, len as i32, glow::MAP_WRITE_BIT | glow::MAP_UNSYNCHRONIZED_BIT | glow::MAP_INVALIDATE_RANGE_BIT);
            if p.is_null() {
                self.gl.bind_buffer(glow::ARRAY_BUFFER, None);
                return false;
            }
            std::ptr::copy_nonoverlapping(shadow.as_ptr(), p, len as usize);
            self.gl.unmap_buffer(glow::ARRAY_BUFFER);
            self.gl.bind_buffer(glow::ARRAY_BUFFER, None);
            Self::point_vao(&self.gl, m.vao, vbo, at);
        }
        true
    }

    /// Point a mesh VAO's four attributes at `buffer` from byte `base` (its element buffer stays).
    unsafe fn point_vao(gl: &glow::Context, vao: glow::VertexArray, buffer: glow::Buffer, base: u64) {
        unsafe {
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(buffer));
            let s = VERTEX_STRIDE as i32;
            for (i, n, off) in [(0u32, 3, 0i32), (1, 3, 12), (2, 2, 24), (3, 4, 32)] {
                gl.vertex_attrib_pointer_f32(i, n, glow::FLOAT, false, s, base as i32 + off);
            }
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
        }
    }

    /// ReNFS5: give arena mesh `id` a vertex buffer, an index buffer and a VAO of its own. The
    /// vertices are copied GPU to GPU and the indices come from the copy kept at upload, so nothing
    /// waits for the GPU (reading the arena back cost 200 ms when a few hundred meshes moved at once).
    fn promote_mesh(&mut self, id: u32) {
        let Some(m) = self.meshes.get_mut(id as usize) else { return };
        let Some(local) = m.local_indices.take() else { return };
        let (vlen, voff, avbo) = (m.vertex_len, m.vertex_offset, m.vbo);
        // SAFETY: GL buffer work on the current context: a copy inside the GPU, then new objects
        unsafe {
            self.gl.bind_vertex_array(None);
            self.st.vao = None;
            let vbo = self.gl.create_buffer().expect("buffer");
            let ibo = self.gl.create_buffer().expect("buffer");
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(vbo));
            self.gl.buffer_data_size(glow::COPY_WRITE_BUFFER, vlen.max(4) as i32, glow::STREAM_DRAW);
            if vlen > 0 {
                self.gl.bind_buffer(glow::COPY_READ_BUFFER, Some(avbo));
                self.gl.copy_buffer_sub_data(glow::COPY_READ_BUFFER, glow::COPY_WRITE_BUFFER, voff as i32, 0, vlen as i32);
                self.gl.bind_buffer(glow::COPY_READ_BUFFER, None);
            }
            self.gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
            self.gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ibo));
            let ib: &[u8] = bytemuck::cast_slice(&local);
            self.gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, if ib.is_empty() { &[0; 4] } else { ib }, glow::STATIC_DRAW);
            let vao = self.make_vao(vbo, ibo);
            let m = &mut self.meshes[id as usize];
            m.vao = vao;
            m.vbo = vbo;
            m.index_offset = 0;
            m.vertex_offset = 0;
            m.dynamic_size = Some(vlen);
        }
    }

    /// The vertices of promoted mesh `id` read back into its kept copy (only for a part-write before
    /// any whole write — waits for the GPU once).
    fn read_back_shadow(&mut self, id: u32) {
        let m = &self.meshes[id as usize];
        let (vbo, vlen) = (m.vbo, m.vertex_len as usize);
        let mut v = vec![0u8; vlen];
        // SAFETY: a range of the mesh's own buffer, mapped for reading and unmapped
        unsafe {
            self.gl.bind_vertex_array(None);
            self.st.vao = None;
            self.gl.bind_buffer(glow::COPY_READ_BUFFER, Some(vbo));
            let p = self.gl.map_buffer_range(glow::COPY_READ_BUFFER, 0, vlen as i32, glow::MAP_READ_BIT);
            if !p.is_null() {
                std::ptr::copy_nonoverlapping(p, v.as_mut_ptr(), vlen);
                self.gl.unmap_buffer(glow::COPY_READ_BUFFER);
            }
            self.gl.bind_buffer(glow::COPY_READ_BUFFER, None);
        }
        self.meshes[id as usize].shadow = Some(v);
    }

    pub fn upload_texture_mips(&mut self, width: u32, height: u32, mips: &[Vec<u8>]) -> TextureId {
        // SAFETY: GL texture work on the current context, on the scratch unit
        unsafe {
            let t = self.gl.create_texture().expect("texture");
            self.bind_unit_texture(U_SCRATCH, Some(t));
            let (mut w, mut h) = (width.max(1), height.max(1));
            let mut levels = 0;
            for data in mips {
                let need = (w * h * 4) as usize;
                if data.len() < need {
                    break;
                }
                self.gl.tex_image_2d(glow::TEXTURE_2D, levels, glow::RGBA8 as i32, w as i32, h as i32, 0, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(Some(&data[..need])));
                levels += 1;
                if w == 1 && h == 1 {
                    break;
                }
                w = (w / 2).max(1);
                h = (h / 2).max(1);
            }
            if levels == 0 {
                self.gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, 1, 1, 0, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(Some(&[255, 255, 255, 255])));
                levels = 1;
            }
            self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAX_LEVEL, levels - 1);
            self.textures.push(GlTex { tex: t, width, height });
        }
        TextureId(self.textures.len() as u32 - 1)
    }

    pub fn upload_textures(&mut self, items: &[(u32, u32, &[Vec<u8>])]) -> Vec<TextureId> {
        items.iter().map(|&(w, h, m)| self.upload_texture_mips(w, h, m)).collect()
    }

    fn update_texture_level(&mut self, id: u32, level: u32, w: u32, h: u32, data: &[u8]) {
        let Some(t) = self.textures.get(id as usize).map(|t| t.tex) else { return };
        if data.len() < (w * h * 4) as usize {
            return;
        }
        // SAFETY: GL texture work on the current context
        unsafe {
            self.bind_unit_texture(U_SCRATCH, Some(t));
            self.gl.tex_sub_image_2d(glow::TEXTURE_2D, level as i32, 0, 0, w as i32, h as i32, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(Some(&data[..(w * h * 4) as usize])));
        }
    }

    fn set_headlight_map(&mut self, map: &[u8]) {
        if map.len() != 128 * 128 {
            return;
        }
        // SAFETY: GL texture work on the current context
        unsafe {
            self.bind_unit_texture(U_SCRATCH, Some(self.night_tex));
            self.gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            self.gl.tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, 128, 128, glow::RED, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(Some(map)));
            self.gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 4);
        }
    }

    fn create_material(&mut self, d: MaterialDesc) -> MaterialId {
        let texture = d.texture.unwrap_or(self.white);
        let env = d.env_texture.unwrap_or(self.white);
        self.materials.push(GlMat {
            texture,
            env,
            sampler: d.clamp[0] as usize | (d.clamp[1] as usize) << 1,
            alpha_blend: d.alpha_blend,
            blend_pair: d.blend_pair.min(3),
            device_states: d.device_states,
            lod_bias: d.lod_bias,
            cull_back: d.cull_back,
            view_cull: d.view_cull,
            depth_bias: d.depth_bias,
            false_sort: d.false_sort,
            depth_test: d.depth_test,
            env_mix: if d.env_texture.is_some() { d.env_mix } else { 0.0 },
            env_variant: d.env_variant.min(2),
            sky_map: d.sky_map,
        });
        MaterialId(self.materials.len() as u32 - 1)
    }

    fn duplicate_material(&mut self, id: MaterialId, texture: TextureId) -> MaterialId {
        let Some(m) = self.materials.get(id.0 as usize).copied() else { return id };
        self.materials.push(GlMat { texture, ..m });
        MaterialId(self.materials.len() as u32 - 1)
    }

    // ---- GL state ----------------------------------------------------------------------------

    fn bind_unit_texture(&mut self, unit: u32, t: Option<glow::Texture>) {
        // SAFETY: GL state on the current context
        unsafe {
            if self.st.active != unit {
                self.gl.active_texture(glow::TEXTURE0 + unit);
                self.st.active = unit;
            }
            if self.st.textures[unit as usize] != t {
                self.gl.bind_texture(glow::TEXTURE_2D, t);
                self.st.textures[unit as usize] = t;
            }
        }
    }

    fn bind_unit(&mut self, unit: u32, t: glow::Texture, s: glow::Sampler) {
        if self.st.textures[unit as usize] != Some(t) {
            self.bind_unit_texture(unit, Some(t));
        }
        if self.st.samplers[unit as usize] != Some(s) {
            // SAFETY: GL state on the current context
            unsafe { self.gl.bind_sampler(unit, Some(s)) };
            self.st.samplers[unit as usize] = Some(s);
        }
    }

    fn apply_raster(&mut self, r: Raster) {
        let old = self.st.raster;
        let gl = &self.gl;
        // SAFETY: GL state on the current context
        unsafe {
            if old.map(|o| o.blend) != Some(r.blend) {
                match r.blend {
                    None => gl.disable(glow::BLEND),
                    Some((s, d, sa, da)) => {
                        if old.is_none_or(|o| o.blend.is_none()) {
                            gl.enable(glow::BLEND);
                        }
                        gl.blend_func_separate(s, d, sa, da);
                    }
                }
            }
            if old.map(|o| o.depth_test) != Some(r.depth_test) {
                if r.depth_test {
                    gl.enable(glow::DEPTH_TEST);
                } else {
                    gl.disable(glow::DEPTH_TEST);
                }
            }
            if r.depth_test && old.map(|o| o.depth_func) != Some(r.depth_func) {
                gl.depth_func(r.depth_func);
            }
            if old.map(|o| o.depth_write) != Some(r.depth_write) {
                gl.depth_mask(r.depth_write);
            }
            if old.map(|o| o.cull) != Some(r.cull) {
                if r.cull == 0 {
                    gl.disable(glow::CULL_FACE);
                } else {
                    if old.is_none_or(|o| o.cull == 0) {
                        gl.enable(glow::CULL_FACE);
                    }
                    gl.cull_face(r.cull);
                }
            }
            if old.map(|o| o.front_face) != Some(r.front_face) {
                gl.front_face(r.front_face);
            }
        }
        // the depth func is kept only while the test is on: a later switch on re-sends it
        self.st.raster = Some(Raster { depth_func: if r.depth_test { r.depth_func } else { old.map_or(0, |o| o.depth_func) }, ..r });
        if !r.depth_test {
            if let Some(x) = self.st.raster.as_mut() {
                x.depth_func = 0;
            }
        }
    }

    fn use_program(&mut self, p: glow::Program) {
        if self.st.program != Some(p) {
            // SAFETY: a linked program of this context
            unsafe { self.gl.use_program(Some(p)) };
            self.st.program = Some(p);
        }
    }

    fn bind_vao(&mut self, v: glow::VertexArray) {
        if self.st.vao != Some(v) {
            // SAFETY: a vertex array of this context
            unsafe { self.gl.bind_vertex_array(Some(v)) };
            self.st.vao = Some(v);
        }
    }

    /// ReNFS5: every program the shader cache holds, loaded now (the first draw) — a cached program
    /// is not compiled mid-race (a Mali compile is 60–80 ms, a visible hitch).
    fn preload_programs(&mut self) {
        let Some(c) = self.shader_cache.as_mut() else { return };
        if c.preloaded {
            return;
        }
        c.preloaded = true;
        let keys = c.keys();
        let t0 = std::time::Instant::now();
        let mut n = 0;
        for key in keys {
            if !self.programs.contains_key(&key) && self.load_cached_program(key) {
                n += 1;
            }
        }
        tracing::info!(programs = n, ms = t0.elapsed().as_millis() as u64, "shader cache loaded");
    }

    /// The program `key` from the shader cache, when it holds one for these sources and this driver.
    fn load_cached_program(&mut self, key: u32) -> bool {
        let (vs, fs) = gl_shaders::mesh_sources(key);
        let Some(c) = self.shader_cache.as_ref() else { return false };
        let Some(bin) = c.load(key, &vs, &fs) else { return false };
        // SAFETY: a program object of this context, given a binary this driver produced
        let p = unsafe {
            let Ok(p) = self.gl.create_program() else { return false };
            self.gl.program_binary(p, &bin);
            if !self.gl.get_program_link_status(p) {
                self.gl.delete_program(p);
                return false;
            }
            p
        };
        self.adopt_program(key, p);
        true
    }

    fn mesh_program(&mut self, key: u32) -> Option<u32> {
        if self.failed_programs.contains(&key) {
            return None;
        }
        self.preload_programs();
        if !self.programs.contains_key(&key) && !self.load_cached_program(key) {
            let (vs, fs) = gl_shaders::mesh_sources(key);
            let p = match compile(&self.gl, &vs, &fs, &[(0, "a_pos"), (1, "a_normal"), (2, "a_uv"), (3, "a_color")]) {
                Ok(p) => p,
                Err(e) => {
                    tracing::error!(key = format_args!("{key:#x}"), error = %e, "mesh program");
                    self.failed_programs.insert(key);
                    return None;
                }
            };
            if let Some(c) = self.shader_cache.as_ref() {
                // SAFETY: the program just linked
                if let Some(bin) = unsafe { self.gl.get_program_binary(p) } {
                    c.save(key, &vs, &fs, &bin);
                }
            }
            tracing::info!(key = format_args!("{key:#x}"), programs = self.programs.len() + 1, "mesh program compiled");
            self.adopt_program(key, p);
        }
        Some(key)
    }

    /// A linked mesh program into `programs`: its samplers bound to their units, its uniforms found.
    fn adopt_program(&mut self, key: u32, p: glow::Program) {
        {
            let gl = &self.gl;
            // SAFETY: the program just linked
            unsafe {
                gl.use_program(Some(p));
                for (name, unit) in [("t_diffuse", U_DIFFUSE), ("t_env", U_ENV), ("t_sky", U_SKY), ("t_proj", U_PROJ), ("t_head", U_HEAD), ("t_night", U_NIGHT)] {
                    if let Some(l) = gl.get_uniform_location(p, name) {
                        gl.uniform_1_i32(Some(&l), unit as i32);
                    }
                }
            }
            self.st.program = Some(p);
            let mp = MeshProg {
                prog: p,
                mvp: Uni::new(gl, p, "u_mvp"),
                model: Uni::new(gl, p, "u_model"),
                view: Uni::new(gl, p, "u_view"),
                headlight: Uni::new(gl, p, "u_headlight"),
                beacon: Uni::new(gl, p, "u_beacon"),
                sun: Uni::new(gl, p, "u_sun"),
                light: Uni::new(gl, p, "u_light"),
                misc: Uni::new(gl, p, "u_misc"),
                misc2: Uni::new(gl, p, "u_misc2"),
                fog: Uni::new(gl, p, "u_fog"),
                lit_a: Uni::new(gl, p, "u_lit_a"),
                lit_d: Uni::new(gl, p, "u_lit_d"),
                lit_dir: Uni::new(gl, p, "u_lit_dir"),
                sky_u: Uni::new(gl, p, "u_sky_u"),
                sky_v: Uni::new(gl, p, "u_sky_v"),
                shadow_u: Uni::new(gl, p, "u_shadow_u"),
                shadow_v: Uni::new(gl, p, "u_shadow_v"),
                pass: u64::MAX,
            };
            self.programs.insert(key, mp);
        }
    }

    // ---- frames ------------------------------------------------------------------------------

    fn depth_rb(&mut self, w: u32, h: u32) -> glow::Renderbuffer {
        if let Some(r) = self.depth_rbs.get(&(w, h)) {
            return *r;
        }
        // SAFETY: GL object creation on the current context
        let rb = unsafe {
            let rb = self.gl.create_renderbuffer().expect("renderbuffer");
            self.gl.bind_renderbuffer(glow::RENDERBUFFER, Some(rb));
            self.gl.renderbuffer_storage(glow::RENDERBUFFER, glow::DEPTH_COMPONENT24, w as i32, h as i32);
            self.gl.bind_renderbuffer(glow::RENDERBUFFER, None);
            rb
        };
        self.depth_rbs.insert((w, h), rb);
        rb
    }

    /// A framebuffer over `tex` (`w × h`) with the shared depth buffer of that size.
    fn make_fbo(&mut self, tex: glow::Texture, w: u32, h: u32) -> glow::Framebuffer {
        let rb = self.depth_rb(w, h);
        // SAFETY: GL object creation on the current context
        unsafe {
            let fbo = self.gl.create_framebuffer().expect("framebuffer");
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            self.gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(tex), 0);
            self.gl.framebuffer_renderbuffer(glow::FRAMEBUFFER, glow::DEPTH_ATTACHMENT, glow::RENDERBUFFER, Some(rb));
            let status = self.gl.check_framebuffer_status(glow::FRAMEBUFFER);
            if status != glow::FRAMEBUFFER_COMPLETE {
                tracing::error!(status = format_args!("{status:#x}"), w, h, "framebuffer incomplete");
            }
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            fbo
        }
    }

    fn colour_texture(&mut self, w: u32, h: u32) -> glow::Texture {
        // SAFETY: GL texture work on the current context
        unsafe {
            let t = self.gl.create_texture().expect("texture");
            self.bind_unit_texture(U_SCRATCH, Some(t));
            self.gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, w as i32, h as i32, 0, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
            self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAX_LEVEL, 0);
            t
        }
    }

    pub fn render(&mut self, scene: &Scene, camera: &Camera) -> Result<()> {
        self.lens_async = true;
        let r = self.render_window(scene, camera);
        self.lens_async = false;
        r
    }

    fn render_window(&mut self, scene: &Scene, camera: &Camera) -> Result<()> {
        let (w, h) = self.size;
        let aspect = w as f32 / h.max(1) as f32;
        let probe = scene.lens_probe.filter(|_| self.lens_enabled);
        if self.gamma == 1.0 {
            self.draw_frame(scene, camera, Tgt { fbo: None, width: w, height: h, flip: false }, scene.clear_color, true, aspect, VIEW_FAR, probe);
        } else {
            let fbo = self.gamma_source(w, h);
            self.draw_frame(scene, camera, Tgt { fbo: Some(fbo), width: w, height: h, flip: false }, scene.clear_color, true, aspect, VIEW_FAR, probe);
            self.gamma_resolve(None, w, h);
        }
        crate::hitch::mark("gl_draw");
        self.frame_dump(w, h);
        self.stream_fence();
        self.window.gl_swap();
        crate::hitch::mark("swap");
        self.stats_draws_report();
        Ok(())
    }

    fn stats_draws_report(&mut self) {
        if self.stats_t.elapsed() >= std::time::Duration::from_secs(5) {
            tracing::info!(draws = self.stats_draws, programs = self.programs.len(), "GLES draws in the last 5 s");
            self.stats_draws = 0;
            self.stats_t = std::time::Instant::now();
        }
    }

    fn gamma_source(&mut self, w: u32, h: u32) -> glow::Framebuffer {
        if let Some((gw, gh, fbo, _)) = self.gamma_target
            && gw == w
            && gh == h
        {
            return fbo;
        }
        let t = self.colour_texture(w, h);
        let fbo = self.make_fbo(t, w, h);
        self.gamma_target = Some((w, h, fbo, t));
        fbo
    }

    /// The display ramp of state 0x2E over the frame `gamma_source` holds, into `dst`.
    fn gamma_resolve(&mut self, dst: Option<glow::Framebuffer>, w: u32, h: u32) {
        let Some((_, _, _, src)) = self.gamma_target else { return };
        let (prog, loc) = (self.gamma_prog.0, self.gamma_prog.1.clone());
        self.apply_raster(Raster { blend: None, depth_test: false, depth_func: glow::LEQUAL, depth_write: false, cull: 0, front_face: glow::CW });
        self.use_program(prog);
        self.bind_unit(0, src, self.nearest_sampler);
        // SAFETY: GL draw on the current context
        unsafe {
            self.gl.uniform_1_f32(loc.as_ref(), self.gamma);
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, dst);
            self.gl.viewport(0, 0, w as i32, h as i32);
            self.bind_vao(self.empty_vao);
            self.gl.draw_arrays(glow::TRIANGLES, 0, 3);
        }
    }

    pub fn render_offscreen(&mut self, scene: &Scene, camera: &Camera, spec: OffscreenSpec) -> TextureId {
        let OffscreenSpec { width, height, reuse, clear_colour, depth_far } = spec;
        let (width, height) = (width.max(1), height.max(1));
        let id = match reuse.filter(|t| self.offscreen.get(&t.0).is_some_and(|(w, h)| *w == width && *h == height)) {
            Some(t) => t,
            None => {
                let tex = self.colour_texture(width, height);
                let fbo = self.make_fbo(tex, width, height);
                self.textures.push(GlTex { tex, width, height });
                let id = TextureId(self.textures.len() as u32 - 1);
                self.offscreen.insert(id.0, (width, height));
                self.offscreen_fbos.insert(id.0, Offscreen { fbo });
                id
            }
        };
        let fbo = self.offscreen_fbos[&id.0].fbo;
        // drawn top row first: the sprite pass samples it with top-left uv
        self.draw_frame(scene, camera, Tgt { fbo: Some(fbo), width, height, flip: true }, clear_colour.unwrap_or([0.0; 4]), false, width as f32 / height as f32, depth_far.unwrap_or(VIEW_FAR), None);
        id
    }

    pub fn render_to_image(&mut self, scene: &Scene, camera: &Camera, width: u32, height: u32) -> Result<Vec<u8>> {
        let tex = self.colour_texture(width, height);
        let fbo = self.make_fbo(tex, width, height);
        let probe = scene.lens_probe;
        if self.gamma == 1.0 {
            self.draw_frame(scene, camera, Tgt { fbo: Some(fbo), width, height, flip: false }, scene.clear_color, true, width as f32 / height as f32, VIEW_FAR, probe);
        } else {
            let src = self.gamma_source(width, height);
            self.draw_frame(scene, camera, Tgt { fbo: Some(src), width, height, flip: false }, scene.clear_color, true, width as f32 / height as f32, VIEW_FAR, probe);
            self.gamma_resolve(Some(fbo), width, height);
        }
        let mut px = vec![0u8; (width * height * 4) as usize];
        // SAFETY: a read of the framebuffer just drawn
        unsafe {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            self.gl.read_pixels(0, 0, width as i32, height as i32, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::Slice(Some(&mut px)));
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.gl.delete_framebuffer(fbo);
            self.gl.delete_texture(tex);
        }
        self.st.textures = [None; UNITS];
        // GL's rows are bottom first
        let row = (width * 4) as usize;
        let mut out = Vec::with_capacity(px.len());
        for y in (0..height as usize).rev() {
            out.extend_from_slice(&px[y * row..(y + 1) * row]);
        }
        Ok(out)
    }

    pub fn render_still(&mut self, scene: &Scene, camera: &Camera, width: u32, height: u32) -> Result<(u32, u32, Vec<u8>)> {
        let (w, h) = still_size(width, height, self.max_tex, 1 << 30);
        let r = self.render_to_image(scene, camera, w, h);
        if let Some(rb) = self.depth_rbs.remove(&(w, h)) {
            // SAFETY: a renderbuffer of this context, no longer attached to a live framebuffer
            unsafe { self.gl.delete_renderbuffer(rb) };
        }
        r.map(|px| (w, h, px))
    }

    /// The lens flare's 4 × 4 block at the probe's point (`gpu.rs`'s `lens_copy`), read now.
    fn lens_read(&mut self, tgt: Tgt, probe: LensProbe, reference: bool) {
        let (w, h) = (tgt.width as i32, tgt.height as i32);
        if w < 5 || h < 5 {
            return;
        }
        let t0 = std::time::Instant::now();
        let (px, py) = ((probe.x / 640.0 * w as f32).round_ties_even() as i32 - 2, (probe.y / 480.0 * h as f32).round_ties_even() as i32 - 2);
        let clamp = |v: i32, size: i32| {
            let v = v.max(0);
            if v + 4 >= size { size - 5 } else { v }
        };
        let (x, y) = (clamp(px, w), clamp(py, h));
        let gl_y = if tgt.flip { y } else { h - y - 4 };
        if self.lens_async {
            // SAFETY: a pixel-pack buffer of this context; the read is queued, not waited for
            unsafe {
                let pbo = match self.lens_pbos.pop() {
                    Some(b) => b,
                    None => {
                        let Ok(b) = self.gl.create_buffer() else { return };
                        self.gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(b));
                        self.gl.buffer_data_size(glow::PIXEL_PACK_BUFFER, 64, glow::STREAM_READ);
                        b
                    }
                };
                self.gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(pbo));
                self.gl.read_pixels(x, gl_y, 4, 4, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::BufferOffset(0));
                self.gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
                match self.gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0) {
                    Ok(fence) => self.lens_pending.push(LensPending { pbo, fence, reference, flip: tgt.flip, pass: self.pass }),
                    Err(_) => self.lens_pbos.push(pbo),
                }
            }
            return;
        }
        let mut block = [0u8; 64];
        // SAFETY: a read of the bound framebuffer
        unsafe { self.gl.read_pixels(x, gl_y, 4, 4, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::Slice(Some(&mut block))) };
        let pixels = lens_block_pixels(&block, tgt.flip);
        self.lens_done.push(LensRead { reference, pixels, cost_ms: t0.elapsed().as_millis() as u32 });
    }

    fn clear(&mut self, colour: Option<[f64; 4]>, depth: bool) {
        // the masks a clear honours
        self.apply_raster(Raster { depth_write: true, ..self.st.raster.unwrap_or(Raster { blend: None, depth_test: true, depth_func: glow::LEQUAL, depth_write: true, cull: 0, front_face: glow::CW }) });
        let mut bits = 0;
        // SAFETY: GL clear on the current context
        unsafe {
            if let Some(c) = colour {
                self.gl.clear_color(c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32);
                bits |= glow::COLOR_BUFFER_BIT;
            }
            if depth {
                self.gl.clear_depth_f32(1.0);
                bits |= glow::DEPTH_BUFFER_BIT;
            }
            self.gl.clear(bits);
        }
    }

    /// One frame of the scene into `tgt`: the 3-D stages, the lens reads, the overlay and the
    /// sprites (`gpu.rs`'s `encode_frame`).
    #[allow(clippy::too_many_arguments)]
    fn draw_frame(&mut self, scene: &Scene, camera: &Camera, tgt: Tgt, clear_colour: [f64; 4], with_sprites: bool, aspect: f32, depth_far: f32, probe: Option<LensProbe>) {
        self.pass += 1;
        let n = scene.items.len().min(MAX_OBJECTS);
        let mut buckets = std::mem::take(&mut self.buckets);
        for b in &mut buckets {
            b.clear();
        }
        for (i, item) in scene.items.iter().take(n).enumerate() {
            let Some(mat) = self.materials.get(item.material.0 as usize) else { continue };
            if item.main_view_only && self.no_reflections {
                continue;
            }
            let stage = if item.overlay { 3 } else if !mat.depth_test { 0 } else if mat.alpha_blend && !item.early { 2 } else { 1 };
            buckets[stage].push(i as u32);
        }
        // SAFETY: GL state on the current context
        unsafe {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, tgt.fbo);
            self.gl.viewport(0, 0, tgt.width as i32, tgt.height as i32);
            self.gl.disable(glow::SCISSOR_TEST);
            self.gl.color_mask(true, true, true, true);
        }
        let vp = camera.view_projection_mirrored(aspect, scene.mirror);
        let vp64 = glam::DMat4::from_cols_array(&vp.to_cols_array().map(f64::from));
        let pass = PassCtx {
            vp64,
            view: camera.view().to_cols_array(),
            sun: [scene.sun_direction.x, scene.sun_direction.y, scene.sun_direction.z, scene.ambient],
            headlight: scene.headlight.unwrap_or(glam::Mat4::IDENTITY).to_cols_array(),
            beacon: scene.beacon.unwrap_or(glam::Mat4::IDENTITY).to_cols_array(),
            flip: if tgt.flip { -1.0 } else { 1.0 },
            // the depth bias's scale (`gpu.rs`): the pass's own planes against the port's
            bias_scale: (camera.far / (camera.far - camera.near)) / (depth_far / (depth_far - camera.near)),
        };
        let probe = probe.filter(|_| tgt.width >= 5);
        if let Some(p) = probe.filter(|p| p.reference) {
            self.clear(Some(clear_colour), true);
            self.draw_bucket(scene, &buckets[0], &pass);
            self.lens_read(tgt, p, true);
        }
        crate::hitch::mark("setup");
        self.clear(Some(clear_colour), true);
        crate::hitch::mark("clear");
        for b in 0..3 {
            self.draw_bucket(scene, &buckets[b], &pass);
            crate::hitch::mark(["bucket0", "bucket1", "bucket2"][b]);
        }
        if let Some(p) = probe.filter(|p| p.test) {
            self.lens_read(tgt, p, false);
        }
        if !buckets[3].is_empty() {
            self.draw_bucket(scene, &buckets[3], &pass);
        }
        crate::hitch::mark("bucket3");
        self.buckets = buckets;
        if with_sprites && !scene.sprites.is_empty() {
            self.draw_sprites(scene, aspect);
        }
        crate::hitch::mark("sprites");
    }

    fn draw_bucket(&mut self, scene: &Scene, bucket: &[u32], pass: &PassCtx) {
        for &idx in bucket {
            let item = &scene.items[idx as usize];
            let mat = self.materials[item.material.0 as usize];
            // mirrored world: the winding flips (`gpu.rs`'s `draw_bucket`)
            let cull = match (mat.cull_back, scene.mirror != item.mirrored, mat.view_cull) {
                (false, _, _) | (true, true, false) => 0,
                (true, false, _) => glow::BACK,
                (true, true, true) => glow::FRONT,
            };
            let sky_tex = if mat.sky_map && !self.no_reflections { item.sky.and_then(|s| s.textures[mat.env_variant]) } else { None };
            let env_on = mat.env_mix > 0.0 && !self.no_reflections;
            let layered = mat.alpha_blend && mat.blend_pair == 0 && (sky_tex.is_some() || item.shadow.is_some() || env_on);
            let blend = if layered {
                Some((glow::ONE, glow::ONE_MINUS_SRC_ALPHA, glow::ONE, glow::ONE_MINUS_SRC_ALPHA))
            } else if mat.alpha_blend {
                Some(match mat.blend_pair {
                    1 => (glow::SRC_ALPHA, glow::ONE, glow::SRC_ALPHA, glow::ONE),
                    2 => (glow::ZERO, glow::ONE_MINUS_SRC_ALPHA, glow::ZERO, glow::ONE_MINUS_SRC_ALPHA),
                    3 => (glow::DST_COLOR, glow::ZERO, glow::DST_COLOR, glow::ZERO),
                    // wgpu's ALPHA_BLENDING: the alpha channel adds `ONE` / `INVSRCALPHA`
                    _ => (glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA, glow::ONE, glow::ONE_MINUS_SRC_ALPHA),
                })
            } else {
                None
            };
            let zwrite_group = mat.depth_test && (!mat.alpha_blend || mat.device_states);
            let zwrite_plain = mat.depth_test && !mat.alpha_blend;
            let fakesort_single = match self.fakesort_mode {
                FakeSortMode::Single => true,
                FakeSortMode::Double => false,
                FakeSortMode::PerItem => !item.fakesort_near,
            };
            let double = mat.false_sort && mat.alpha_blend && !fakesort_single;
            let point = mat.false_sort && mat.alpha_blend && fakesort_single;
            // the flip of an image kept top row first reverses the winding on screen
            let front_face = if pass.flip < 0.0 { glow::CCW } else { glow::CW };
            let raster = |zwrite: bool| Raster { blend, depth_test: mat.depth_test, depth_func: glow::LEQUAL, depth_write: zwrite, cull, front_face };
            let alpha_ref = if mat.device_states || item.alpha_test { item.alpha_ref as f32 + 0.5 } else { 0.0 };
            let fog = if item.sky_pass { scene.sky_fog } else { scene.fog };
            let headlight_on = scene.headlight.is_some();
            let mut key = if layered { 2 } else { mat.alpha_blend as u32 };
            if alpha_ref > 0.0 && !layered {
                key |= f::ALPHA_TEST;
            }
            if item.headlit && headlight_on {
                key |= f::HEADLIGHT;
            }
            if item.headlit && scene.beacon.is_some() {
                key |= f::BEACON;
            }
            if item.uv_lit && headlight_on && scene.headlight_decal {
                key |= f::UVLIT;
            }
            if item.shadow.is_some() {
                key |= f::SHADOW;
            }
            if sky_tex.is_some() {
                key |= f::SKY;
            }
            if env_on {
                key |= f::ENV;
            }
            if item.layer_light.is_some() {
                key |= f::LAYERLIGHT;
            }
            if fog.is_some_and(|f| f.density > 0.0) {
                key |= f::FOG;
            }
            let Some(key) = self.mesh_program(key) else { continue };
            let prog = self.programs[&key].prog;
            self.use_program(prog);
            // textures
            let tex = self.textures[mat.texture.0 as usize].tex;
            let s = if point { self.point_samplers[mat.sampler] } else { self.samplers[mat.sampler | (!self.trilinear as usize) << 2] };
            self.bind_unit(U_DIFFUSE, tex, s);
            let ps = self.proj_samplers[!self.trilinear as usize];
            if key & f::ENV != 0 {
                let t = self.textures[mat.env.0 as usize].tex;
                self.bind_unit(U_ENV, t, self.env_sampler);
            }
            if let Some(t) = sky_tex {
                let t = self.textures.get(t.0 as usize).map_or(self.textures[self.white.0 as usize].tex, |t| t.tex);
                self.bind_unit(U_SKY, t, ps);
            }
            if key & f::SHADOW != 0 {
                let t = self.proj_tex.and_then(|t| self.textures.get(t.0 as usize)).map_or(self.proj_default, |t| t.tex);
                self.bind_unit(U_PROJ, t, ps);
            }
            if key & f::UVLIT != 0 {
                let t = self.head_tex.and_then(|t| self.textures.get(t.0 as usize)).map_or(self.proj_default, |t| t.tex);
                self.bind_unit(U_HEAD, t, ps);
            }
            if key & (f::HEADLIGHT | f::BEACON | f::UVLIT) != 0 {
                self.bind_unit(U_NIGHT, self.night_tex, self.nearest_sampler);
            }
            // uniforms
            let gl = &self.gl;
            let p = self.programs.get_mut(&key).expect("program");
            if p.pass != self.pass {
                p.pass = self.pass;
                set16(gl, &mut p.view, pass.view);
                set4(gl, &mut p.sun, pass.sun);
                set16(gl, &mut p.headlight, pass.headlight);
                set16(gl, &mut p.beacon, pass.beacon);
            }
            let model = item.transform.to_cols_array();
            let mvp = (pass.vp64 * glam::DMat4::from_cols_array(&model.map(f64::from))).as_mat4().to_cols_array();
            set16(gl, &mut p.mvp, mvp);
            set16(gl, &mut p.model, model);
            set4(gl, &mut p.light, [item.light[0], item.light[1], item.light[2], item.alpha]);
            set4(gl, &mut p.misc, [item.uv_offset[0], item.uv_offset[1], mat.lod_bias, alpha_ref]);
            let pull = depth_bias_offset(mat.depth_bias.saturating_add(item.depth_bias)) * pass.bias_scale;
            set4(gl, &mut p.misc2, [pull, item.fog_w, pass.flip, 0.0]);
            if let Some(fg) = fog {
                set4(gl, &mut p.fog, [fg.colour[0], fg.colour[1], fg.colour[2], fg.density]);
            }
            if let Some(l) = item.layer_light {
                set4(gl, &mut p.lit_a, [l.ambient[0], l.ambient[1], l.ambient[2], 1.0]);
                set4(gl, &mut p.lit_d, [l.diffuse[0], l.diffuse[1], l.diffuse[2], 0.0]);
                set4(gl, &mut p.lit_dir, [l.direction[0], l.direction[1], l.direction[2], 0.0]);
            }
            if let Some(sk) = item.sky {
                set4(gl, &mut p.sky_u, sk.u);
                set4(gl, &mut p.sky_v, sk.v);
            }
            if let Some((u, v)) = item.shadow {
                set4(gl, &mut p.shadow_u, u);
                set4(gl, &mut p.shadow_v, v);
            }
            let Some(mesh) = self.meshes.get(item.mesh.0 as usize) else { continue };
            let (vao, count, offset) = (mesh.vao, mesh.index_count, mesh.index_offset);
            if count == 0 {
                continue;
            }
            self.bind_vao(vao);
            let mut draw = |core: &mut Self, zwrite: bool| {
                core.apply_raster(raster(zwrite));
                // SAFETY: the bound vertex array's element buffer holds `count` indices at `offset`
                unsafe { core.gl.draw_elements(glow::TRIANGLES, count as i32, glow::UNSIGNED_INT, offset as i32) };
                core.stats_draws += 1;
            };
            if double {
                // `LayerFalseSort1` / `2`: depth writes off, then the material's own
                draw(self, false);
                draw(self, zwrite_group);
            } else if layered {
                draw(self, mat.depth_test && mat.device_states);
            } else if mat.alpha_blend {
                draw(self, if mat.device_states { zwrite_group } else { zwrite_plain });
            } else {
                draw(self, zwrite_plain);
            }
        }
    }

    fn draw_sprites(&mut self, scene: &Scene, aspect: f32) {
        let n = scene.sprites.len().min(MAX_SPRITES);
        let mut verts = std::mem::take(&mut self.sprite_scratch);
        verts.clear();
        verts.reserve(n * 6);
        for sp in scene.sprites.iter().take(n) {
            verts.extend_from_slice(&sprite_quad(sp, aspect));
        }
        // SAFETY: GL buffer work on the current context
        unsafe {
            self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.sprite_vbo));
            self.gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck::cast_slice(&verts), glow::STREAM_DRAW);
            self.gl.bind_buffer(glow::ARRAY_BUFFER, None);
        }
        self.sprite_scratch = verts;
        // the 2D pass starts with the depth cleared to 1.0 (`gpu.rs`)
        self.clear(None, true);
        self.bind_vao(self.sprite_vao);
        let mut first = 0usize;
        while first < n {
            let sp = &scene.sprites[first];
            let compare = sp.depth.map(|d| d.compare);
            let tex = sp.texture.0;
            let mut end = first + 1;
            while end < n && scene.sprites[end].texture.0 == tex && scene.sprites[end].depth.map(|d| d.compare) == compare {
                end += 1;
            }
            let Some(t) = self.textures.get(tex as usize).map(|t| t.tex) else {
                first = end;
                continue;
            };
            let raster = match compare {
                None => Raster { blend: Some((glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA, glow::ONE, glow::ONE_MINUS_SRC_ALPHA)), depth_test: false, depth_func: glow::LEQUAL, depth_write: false, cull: 0, front_face: glow::CW },
                Some(c) => Raster {
                    blend: Some((glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA, glow::ONE, glow::ONE_MINUS_SRC_ALPHA)),
                    depth_test: true,
                    depth_func: if c == SpriteCompare::LessEqual { glow::LEQUAL } else { glow::GEQUAL },
                    depth_write: true,
                    cull: 0,
                    front_face: glow::CW,
                },
            };
            self.apply_raster(raster);
            let prog = self.sprite_progs[compare.is_some() as usize].prog;
            self.use_program(prog);
            self.bind_unit(0, t, self.sprite_sampler);
            // SAFETY: the sprite stream holds `n · 6` vertices
            unsafe { self.gl.draw_arrays(glow::TRIANGLES, (first * 6) as i32, ((end - first) * 6) as i32) };
            self.stats_draws += 1;
            first = end;
        }
    }
}

struct PassCtx {
    vp64: glam::DMat4,
    view: [f32; 16],
    sun: [f32; 4],
    headlight: [f32; 16],
    beacon: [f32; 16],
    flip: f32,
    bias_scale: f32,
}

/// The two triangles of a sprite in clip space (`gpu.rs`'s `sprite_quad`).
fn sprite_quad(sp: &Sprite, aspect: f32) -> [SpriteVertex; 6] {
    let (z, alpha_ref) = sp.depth.map_or((0.0, 0.0), |d| (d.z, d.alpha_test.map_or(-1.0, f32::from)));
    if let Some((t, col)) = sp.tri {
        let v = |p: [f32; 2], c: [f32; 4]| SpriteVertex { pos: [p[0] * 2.0 - 1.0, 1.0 - p[1] * 2.0, z], uv: sp.uv_min, color: c, alpha_ref };
        let (a, b, c) = (v(t[0], col[0]), v(t[1], col[1]), v(t[2], col[2]));
        return [a, b, c, c, c, c];
    }
    let (sx, sy) = (sp.size[0] / aspect, sp.size[1]);
    let (px, py) = (sp.pos[0] + sp.pivot[0] * sx, sp.pos[1] + sp.pivot[1] * sy);
    let (sn, cs) = (sp.rotation * std::f32::consts::TAU).sin_cos();
    let corner = |u: f32, v: f32| -> SpriteVertex {
        let lx = (u - sp.pivot[0]) * sx * aspect;
        let ly = (v - sp.pivot[1]) * sy;
        let rx = lx * cs - ly * sn;
        let ry = lx * sn + ly * cs;
        let x = px + rx / aspect;
        let y = py + ry;
        SpriteVertex {
            pos: [x * 2.0 - 1.0, 1.0 - y * 2.0, z],
            uv: [sp.uv_min[0] + (sp.uv_max[0] - sp.uv_min[0]) * u, sp.uv_min[1] + (sp.uv_max[1] - sp.uv_min[1]) * v],
            color: sp.corner_colors.map_or(sp.color, |k| match (u > 0.5, v > 0.5) {
                (false, false) => k[0],
                (true, false) => k[1],
                (true, true) => k[2],
                (false, true) => k[3],
            }),
            alpha_ref,
        }
    };
    let (a, b, c, d) = (corner(0.0, 0.0), corner(1.0, 0.0), corner(1.0, 1.0), corner(0.0, 1.0));
    [a, b, c, a, c, d]
}

/// ReNFS5: a lens read queued into a pixel-pack buffer (`GlCore::lens_read`).
struct LensPending {
    pbo: glow::Buffer,
    fence: glow::Fence,
    reference: bool,
    flip: bool,
    pass: u64,
}

/// The 4 × 4 RGBA block of a lens read as the sixteen texels, top row first.
fn lens_block_pixels(block: &[u8; 64], flip: bool) -> [u32; 16] {
    let mut pixels = [0u32; 16];
    for row in 0..4 {
        let src = if flip { row } else { 3 - row };
        for i in 0..4 {
            let o = src * 16 + i * 4;
            pixels[row * 4 + i] = u32::from_le_bytes([block[o], block[o + 1], block[o + 2], block[o + 3]]);
        }
    }
    pixels
}

/// ReNFS5: the vertex stream: one buffer in three per-frame thirds, each fenced at its frame's end.
const STREAM_SEGS: usize = 3;
const STREAM_SEG_SIZE: u64 = 12 << 20;

struct StreamRing {
    vbo: glow::Buffer,
    seg_size: u64,
    seg: usize,
    cursor: u64,
    fences: [Option<glow::Fence>; STREAM_SEGS],
    /// the meshes whose vertices each third holds (a later write elsewhere moves them)
    residents: [Vec<u32>; STREAM_SEGS],
}

/// `NAME=0` in the environment (test switches).
fn env_off(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v.trim() == "0")
}

/// ReNFS5: the mesh programs the driver compiled, kept on disk (`NFS5_SHADER_CACHE`, the port puts it
/// in `configs/shader-cache`) so a later run loads them instead of compiling mid-race. A file holds
/// one program, tagged with a hash of the driver's identity and the program's sources: another
/// driver or another build's shaders do not match and are compiled again (and the file replaced).
struct ShaderCache {
    dir: std::path::PathBuf,
    driver: String,
    preloaded: bool,
}

const SHADER_CACHE_MAGIC: &[u8; 8] = b"NFS5PB1\0";

impl ShaderCache {
    fn open(renderer: &str, version: &str) -> Option<Self> {
        let dir = std::path::PathBuf::from(std::env::var_os("NFS5_SHADER_CACHE").filter(|v| !v.is_empty())?);
        std::fs::create_dir_all(&dir).ok()?;
        Some(Self { dir, driver: format!("{renderer}|{version}"), preloaded: false })
    }

    fn path(&self, key: u32) -> std::path::PathBuf {
        self.dir.join(format!("mesh-{key:08x}.bin"))
    }

    fn tag(&self, vs: &str, fs: &str) -> u64 {
        // FNV-1a over the driver and the sources
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in self.driver.bytes().chain([0u8]).chain(vs.bytes()).chain([0u8]).chain(fs.bytes()) {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    fn keys(&self) -> Vec<u32> {
        let Ok(rd) = std::fs::read_dir(&self.dir) else { return Vec::new() };
        rd.filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            u32::from_str_radix(name.strip_prefix("mesh-")?.strip_suffix(".bin")?, 16).ok()
        })
        .collect()
    }

    fn load(&self, key: u32, vs: &str, fs: &str) -> Option<glow::ProgramBinary> {
        let d = std::fs::read(self.path(key)).ok()?;
        if d.len() < 24 || &d[..8] != SHADER_CACHE_MAGIC || u64::from_le_bytes(d[8..16].try_into().ok()?) != self.tag(vs, fs) {
            return None;
        }
        let format = u32::from_le_bytes(d[16..20].try_into().ok()?);
        let len = u32::from_le_bytes(d[20..24].try_into().ok()?) as usize;
        let buffer = d.get(24..24 + len)?.to_vec();
        Some(glow::ProgramBinary { buffer, format })
    }

    fn save(&self, key: u32, vs: &str, fs: &str, bin: &glow::ProgramBinary) {
        if bin.buffer.is_empty() {
            return;
        }
        let mut out = Vec::with_capacity(24 + bin.buffer.len());
        out.extend_from_slice(SHADER_CACHE_MAGIC);
        out.extend_from_slice(&self.tag(vs, fs).to_le_bytes());
        out.extend_from_slice(&bin.format.to_le_bytes());
        out.extend_from_slice(&(bin.buffer.len() as u32).to_le_bytes());
        out.extend_from_slice(&bin.buffer);
        // written beside and renamed: a run cut off mid-write leaves no torn file
        let tmp = self.path(key).with_extension("tmp");
        if std::fs::write(&tmp, &out).is_ok() {
            let _ = std::fs::rename(&tmp, self.path(key));
        }
    }
}
