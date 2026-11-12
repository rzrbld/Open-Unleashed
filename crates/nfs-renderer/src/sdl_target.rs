//! ReNFS5 PLATFORM GLUE: the window's image through SDL2, on the handhelds' two graphics APIs.
//!
//! * **GLES** (the default; every firmware has it, the Mali blob has nothing else): SDL2 makes the
//!   window's EGL context, wgpu runs on that context (`hal::gles::Adapter::new_external`), each
//!   frame is drawn into an image of ours and copied (`glBlitFramebuffer`) onto the window's back
//!   buffer, which `SDL_GL_SwapWindow` shows. wgpu's own GL surface needs an X11 / Wayland window,
//!   which a KMSDRM firmware does not have.
//! * **Vulkan** (Panfrost/panvk and other Mesa firmwares, `NFS_GFX=vulkan`): SDL2 makes the
//!   `VkSurfaceKHR` (`VK_KHR_display` on KMSDRM) on wgpu's instance and wgpu takes it as the
//!   window's surface (`hal::vulkan::set_external_surface`, our wgpu-hal patch). When the window
//!   or the adapter cannot be had, the window is made again for GLES.
//!
//! The GL context is current on one thread at a time: the core is handed over as a whole
//! (`front.rs`'s render thread makes it current on its side, see [`Target::make_current`]).

use anyhow::{Context, Result};
use std::sync::Arc;
use winit::sdl::GfxApi;
use winit::window::Window;

pub(crate) enum Target {
    Surface(wgpu::Surface<'static>),
    Gl(GlTarget),
}

pub(crate) struct GlTarget {
    window: Arc<Window>,
    texture: Option<wgpu::Texture>,
    fbo: Option<glow::Framebuffer>,
}

/// What `Core::new` gets: the instance kept alive, the target, the adapter.
pub(crate) struct Opened {
    pub instance: wgpu::Instance,
    pub target: Target,
    pub adapter: wgpu::Adapter,
}

pub(crate) fn open(window: &Arc<Window>) -> Result<Opened> {
    if window.gfx_api() == GfxApi::Vulkan {
        match open_vulkan(window) {
            Ok(o) => return Ok(o),
            Err(e) => {
                tracing::warn!(error = format!("{e:#}"), "Vulkan unavailable for the window: using GLES");
                window.recreate_for(GfxApi::Gles).context("GLES window")?;
            }
        }
    }
    open_gles(window)
}

fn open_vulkan(window: &Arc<Window>) -> Result<Opened> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::from_build_config().with_env(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let needed = window.vulkan_instance_extensions().context("SDL2 Vulkan extensions")?;
    // SAFETY: the hal instance outlives this block (owned by `instance`)
    let raw = unsafe {
        let hal = instance.as_hal::<wgpu::hal::api::Vulkan>().context("no Vulkan instance (no driver / loader)")?;
        for n in &needed {
            if !hal.shared_instance().extensions().iter().any(|e| *e == n.as_c_str()) {
                tracing::warn!(extension = ?n, "SDL2 needs a Vulkan instance extension wgpu did not enable");
            }
        }
        use ash::vk::Handle;
        hal.shared_instance().raw_instance().handle().as_raw()
    };
    // SAFETY: `raw` is the live instance made with the extensions above
    let vk_surface = unsafe { window.vulkan_create_surface(raw) }.context("SDL_Vulkan_CreateSurface")?;
    wgpu::hal::vulkan::set_external_surface(vk_surface);
    // the handles are not read: the hal instance takes the surface handed over just above
    let dummy = wgpu::SurfaceTargetUnsafe::RawHandle {
        raw_display_handle: Some(raw_window_handle::RawDisplayHandle::Drm(raw_window_handle::DrmDisplayHandle::new(-1))),
        raw_window_handle: raw_window_handle::RawWindowHandle::Drm(raw_window_handle::DrmWindowHandle::new(0)),
    };
    // SAFETY: see above; the surface lives as long as the window, which the renderer keeps
    let surface = unsafe { instance.create_surface_unsafe(dummy) }.context("wgpu surface from SDL2")?;
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .context("no Vulkan adapter for the window")?;
    Ok(Opened { instance, target: Target::Surface(surface), adapter })
}

fn open_gles(window: &Arc<Window>) -> Result<Opened> {
    window.gl_context().context("SDL2 GLES context")?;
    let w = window.clone();
    // SAFETY: the context is current on this thread (made just above)
    let exposed = unsafe { wgpu::hal::gles::Adapter::new_external(|name| w.gl_get_proc_address(name), wgpu::GlBackendOptions::from_env_or_default()) }
        .context("wgpu could not use the GLES context (GLES 3.0 or newer needed)")?;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::empty(), ..wgpu::InstanceDescriptor::new_without_display_handle() });
    // SAFETY: the exposed adapter's context is the window's, current here
    let adapter = unsafe { instance.create_adapter_from_hal(exposed) };
    Ok(Opened { instance, target: Target::Gl(GlTarget { window: window.clone(), texture: None, fbo: None }), adapter })
}

/// The frame being drawn: the surface's image, or our GL image.
pub(crate) enum Frame {
    Surface(wgpu::SurfaceTexture),
    Gl(wgpu::Texture),
}

impl Frame {
    pub fn texture(&self) -> &wgpu::Texture {
        match self {
            Frame::Surface(f) => &f.texture,
            Frame::Gl(t) => t,
        }
    }
}

pub(crate) enum Acquired {
    Frame(Frame),
    /// lost / outdated (reconfigured) or occluded / timed out: skip this frame
    Skip,
}

impl Target {
    pub fn is_gl(&self) -> bool {
        matches!(self, Target::Gl(_))
    }

    /// The image format: the surface's byte format, or RGBA8 for our GL image.
    pub fn format(&self, adapter: &wgpu::Adapter) -> (wgpu::TextureFormat, wgpu::CompositeAlphaMode) {
        match self {
            Target::Surface(s) => {
                let caps = s.get_capabilities(adapter);
                // a byte surface, as Direct3D 7's was (`RE/RENDERING.md`, "Colour space")
                let format = caps
                    .formats
                    .iter()
                    .copied()
                    .find(|f| matches!(f, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm))
                    .or_else(|| caps.formats.iter().copied().find(|f| !f.is_srgb()))
                    .unwrap_or(caps.formats[0]);
                // FIFO is always there; Immediate / Mailbox when the driver has them (`present_mode`)
                (format, caps.alpha_modes[0])
            }
            Target::Gl(_) => (wgpu::TextureFormat::Rgba8Unorm, wgpu::CompositeAlphaMode::Opaque),
        }
    }

    /// The surface's present mode: FIFO with `NFS_VSYNC=1`, else the fastest the driver offers.
    pub fn present_mode(&self, adapter: &wgpu::Adapter) -> wgpu::PresentMode {
        let Target::Surface(s) = self else { return wgpu::PresentMode::AutoNoVsync };
        if std::env::var("NFS_VSYNC").is_ok_and(|v| v.trim() == "1") {
            return wgpu::PresentMode::Fifo;
        }
        let modes = s.get_capabilities(adapter).present_modes;
        [wgpu::PresentMode::Mailbox, wgpu::PresentMode::Immediate].into_iter().find(|m| modes.contains(m)).unwrap_or(wgpu::PresentMode::Fifo)
    }

    pub fn configure(&mut self, device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) {
        match self {
            Target::Surface(s) => s.configure(device, config),
            Target::Gl(g) => {
                g.texture = Some(device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("window image (GLES)"),
                    size: wgpu::Extent3d { width: config.width.max(1), height: config.height.max(1), depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: config.format,
                    usage: config.usage | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                }));
            }
        }
    }

    pub fn acquire(&mut self, device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Result<Acquired> {
        match self {
            Target::Surface(s) => Ok(match s.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => Acquired::Frame(Frame::Surface(f)),
                wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                    s.configure(device, config);
                    Acquired::Skip
                }
                wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => Acquired::Skip,
                other => anyhow::bail!("surface texture: {other:?}"),
            }),
            Target::Gl(g) => {
                if g.texture.is_none() {
                    self.configure(device, config);
                }
                let Target::Gl(g) = self else { unreachable!() };
                Ok(Acquired::Frame(Frame::Gl(g.texture.clone().expect("GL image"))))
            }
        }
    }

    /// Show the frame: the surface's present, or the copy onto the GL back buffer and the swap.
    /// The frame's commands were submitted before (on GL, `submit` runs them on the context).
    pub fn present(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, frame: Frame) {
        match (self, frame) {
            (Target::Surface(_), Frame::Surface(f)) => queue.present(f),
            (Target::Gl(g), Frame::Gl(tex)) => {
                let (w, h) = (tex.width() as i32, tex.height() as i32);
                // SAFETY: GL objects of this context, used on the thread it is current on
                unsafe {
                    let Some(hal_tex) = tex.as_hal::<wgpu::hal::api::Gles>() else { return };
                    // a render-attachment-only image is a renderbuffer in wgpu's GL backend
                    let (rb, raw) = match hal_tex.inner {
                        wgpu::hal::gles::TextureInner::Texture { raw, .. } => (None, Some(raw)),
                        wgpu::hal::gles::TextureInner::Renderbuffer { raw } => (Some(raw), None),
                        _ => {
                            tracing::error!("GL window image of an unexpected kind");
                            return;
                        }
                    };
                    let Some(hal_dev) = device.as_hal::<wgpu::hal::api::Gles>() else { return };
                    let gl = hal_dev.context().lock();
                    use glow::HasContext;
                    let fbo = match g.fbo {
                        Some(f) => f,
                        None => match gl.create_framebuffer() {
                            Ok(f) => *g.fbo.insert(f),
                            Err(e) => {
                                tracing::error!(error = %e, "GL framebuffer");
                                return;
                            }
                        },
                    };
                    gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(fbo));
                    match (rb, raw) {
                        (Some(rb), _) => gl.framebuffer_renderbuffer(glow::READ_FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::RENDERBUFFER, Some(rb)),
                        (_, raw) => gl.framebuffer_texture_2d(glow::READ_FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, raw, 0),
                    }
                    gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
                    gl.disable(glow::SCISSOR_TEST);
                    gl.color_mask(true, true, true, true);
                    // wgpu's GL images are upside down (its shaders flip y): the copy flips them back,
                    // as wgpu's own GL surface does
                    // the window's own size: a frame drawn smaller (`NFS_RENDER_SCALE`) is scaled up here
                    let win = g.window.inner_size();
                    let (ww, wh) = (win.width as i32, win.height as i32);
                    let filter = if (ww, wh) == (w, h) { glow::NEAREST } else { glow::LINEAR };
                    gl.blit_framebuffer(0, h, w, 0, 0, 0, ww, wh, glow::COLOR_BUFFER_BIT, filter);
                    gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
                }
                g.window.gl_swap();
            }
            _ => unreachable!("frame of another target"),
        }
    }

    /// GL: write `bytes` into `buffer` at `offset` now, with `glBufferSubData` (no staging copy). The
    /// caller makes sure no submitted work still reads that range. `false` when not on GL (or the
    /// buffer has no GL name) — the caller then writes it the wgpu way.
    pub fn gl_write_buffer(&self, device: &wgpu::Device, buffer: &wgpu::Buffer, offset: u64, bytes: &[u8]) -> bool {
        if !self.is_gl() || bytes.is_empty() {
            return false;
        }
        // SAFETY: GL objects of this context, on the thread it is current on
        unsafe {
            let Some(raw) = buffer.as_hal::<wgpu::hal::api::Gles>().and_then(|b| b.raw_gl()) else { return false };
            let Some(hal_dev) = device.as_hal::<wgpu::hal::api::Gles>() else { return false };
            let gl = hal_dev.context().lock();
            use glow::HasContext;
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(raw));
            gl.buffer_sub_data_u8_slice(glow::COPY_WRITE_BUFFER, offset as i32, bytes);
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
        }
        true
    }

    /// GL: make the context current on this thread (`true`) or let it go (`false`), for the hand-over
    /// of the core to / from the render thread. Nothing on Vulkan.
    pub fn make_current(&self, current: bool) {
        if let Target::Gl(g) = self {
            g.window.gl_make_current(current);
        }
    }
}
