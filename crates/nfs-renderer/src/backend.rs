//! ReNFS5: the renderer core the front (`front.rs`) drives — the native GLES one (`gl_core.rs`,
//! every GLES firmware; the default) or, for `NFS_GFX=vulkan` on a firmware with a Vulkan driver,
//! the wgpu one (`gpu.rs`). `NFS_RENDERER=wgpu` keeps wgpu on GLES too (comparison only).

use crate::front::CmdList;
use crate::gl_core::GlCore;
use crate::gpu::{MeshId, TextureId};
use crate::{Camera, LensRead, MeshData, Scene};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use winit::sdl::GfxApi;
use winit::window::Window;

pub(crate) enum Core {
    Gl(GlCore),
    Wgpu(crate::gpu::Core),
}

macro_rules! both {
    ($self:ident, $c:ident => $e:expr) => {
        match $self {
            Core::Gl($c) => $e,
            Core::Wgpu($c) => $e,
        }
    };
}

impl Core {
    pub fn new(window: Arc<Window>) -> Result<Self> {
        let wgpu_forced = std::env::var("NFS_RENDERER").is_ok_and(|v| v.trim().eq_ignore_ascii_case("wgpu"));
        if window.gfx_api() == GfxApi::Vulkan || wgpu_forced {
            let core = crate::gpu::Core::new(window.clone())?;
            // a Vulkan window that fell back to GLES (no usable Vulkan): the native core draws it
            if core.is_gl() && !wgpu_forced {
                drop(core);
                return Ok(Core::Gl(GlCore::new(window)?));
            }
            return Ok(Core::Wgpu(core));
        }
        Ok(Core::Gl(GlCore::new(window)?))
    }

    pub fn size(&self) -> (u32, u32) {
        both!(self, c => c.size())
    }
    pub(crate) fn mesh_infos(&self) -> Vec<(u64, u32)> {
        both!(self, c => c.mesh_infos())
    }
    pub(crate) fn texture_count(&self) -> u32 {
        both!(self, c => c.texture_count())
    }
    pub(crate) fn white(&self) -> TextureId {
        both!(self, c => c.white())
    }
    pub(crate) fn material_textures(&self) -> Vec<TextureId> {
        both!(self, c => c.material_textures())
    }
    pub(crate) fn offscreen_sizes(&self) -> HashMap<u32, (u32, u32)> {
        both!(self, c => c.offscreen_sizes())
    }
    pub(crate) fn gl_current(&self, current: bool) {
        both!(self, c => c.gl_current(current))
    }
    pub(crate) fn frame_begin(&mut self) {
        both!(self, c => c.frame_begin())
    }
    pub(crate) fn apply(&mut self, list: &mut CmdList) {
        both!(self, c => c.apply(list))
    }
    pub fn render(&mut self, scene: &Scene, camera: &Camera) -> Result<()> {
        both!(self, c => c.render(scene, camera))
    }
    pub(crate) fn take_lens(&mut self) -> Vec<LensRead> {
        both!(self, c => c.take_lens())
    }
    pub fn upload_mesh(&mut self, mesh: &MeshData) -> MeshId {
        both!(self, c => c.upload_mesh(mesh))
    }
    pub(crate) fn mesh_vertex_len(&self, id: MeshId) -> u64 {
        both!(self, c => c.mesh_vertex_len(id))
    }
    pub fn upload_texture_mips(&mut self, width: u32, height: u32, mips: &[Vec<u8>]) -> TextureId {
        both!(self, c => c.upload_texture_mips(width, height, mips))
    }
    pub fn upload_textures(&mut self, items: &[(u32, u32, &[Vec<u8>])]) -> Vec<TextureId> {
        both!(self, c => c.upload_textures(items))
    }
    pub fn render_to_image(&mut self, scene: &Scene, camera: &Camera, width: u32, height: u32) -> Result<Vec<u8>> {
        both!(self, c => c.render_to_image(scene, camera, width, height))
    }
    pub fn render_still(&mut self, scene: &Scene, camera: &Camera, width: u32, height: u32) -> Result<(u32, u32, Vec<u8>)> {
        both!(self, c => c.render_still(scene, camera, width, height))
    }
}
