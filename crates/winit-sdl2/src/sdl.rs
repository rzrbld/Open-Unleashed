//! The SDL2 side: the window, the event pump and the graphics context (GLES or Vulkan).

use crate::application::ApplicationHandler;
use crate::dpi::{PhysicalPosition, PhysicalSize, Size};
use crate::error::OsError;
use crate::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, MouseScrollDelta, TouchPhase, WindowEvent};
use crate::event_loop::{ActiveEventLoop, ControlFlow};
use crate::monitor::{MonitorHandle, VideoModeHandle};
use crate::window::{WindowAttributes, WindowId};
use sdl2_sys as sys;
use std::ffi::{CStr, CString, c_void};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// The graphics API a window is made for.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GfxApi {
    /// OpenGL ES 3 through SDL2's EGL (every handheld driver: Mali blob, Panfrost, PowerVR…)
    Gles,
    /// Vulkan through SDL2's surface (Panfrost/panvk, Turnip and other Mesa firmwares)
    Vulkan,
}

impl GfxApi {
    /// `NFS_GFX=gles|vulkan` (the launcher's choice), else `WGPU_BACKEND=gl|vulkan`; GLES by default.
    pub fn from_env() -> GfxApi {
        let pick = |v: &str| match v.trim().to_ascii_lowercase().as_str() {
            "vulkan" | "vk" => Some(GfxApi::Vulkan),
            "gles" | "gl" | "opengl" | "es" => Some(GfxApi::Gles),
            _ => None,
        };
        std::env::var("NFS_GFX").ok().and_then(|v| pick(&v)).or_else(|| std::env::var("WGPU_BACKEND").ok().and_then(|v| pick(&v))).unwrap_or(GfxApi::Gles)
    }
}

pub(crate) fn sdl_error() -> String {
    // SAFETY: SDL_GetError returns a valid C string (possibly empty)
    unsafe { CStr::from_ptr(sys::SDL_GetError()) }.to_string_lossy().into_owned()
}

fn hint(name: &str, value: &str) {
    let (n, v) = (CString::new(name).unwrap(), CString::new(value).unwrap());
    // SAFETY: two valid C strings
    unsafe { sys::SDL_SetHint(n.as_ptr(), v.as_ptr()) };
}

pub(crate) fn init_video() -> Result<(), OsError> {
    // the controller code owns SIGINT/SIGTERM handling as before; the game reads Esc / its menus
    hint("SDL_NO_SIGNAL_HANDLERS", "1");
    // on desktop Linux, take the window's GL context through EGL like the handhelds do
    if std::env::var_os("SDL_VIDEO_X11_FORCE_EGL").is_none() {
        hint("SDL_VIDEO_X11_FORCE_EGL", "1");
    }
    // SAFETY: plain SDL initialisation
    if unsafe { sys::SDL_InitSubSystem(sys::SDL_INIT_VIDEO | sys::SDL_INIT_EVENTS) } != 0 {
        return Err(OsError(sdl_error()));
    }
    tracing::info!(driver = video_driver(), "SDL2 video");
    Ok(())
}

/// SDL2's video driver: `KMSDRM`, `mali`, `x11`, `wayland`…
pub fn video_driver() -> String {
    // SAFETY: NULL or a static C string
    let p = unsafe { sys::SDL_GetCurrentVideoDriver() };
    if p.is_null() { String::new() } else { unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned() }
}

/// Fullscreen on a handheld (no window system), windowed on a desktop; `NFS_FULLSCREEN=0/1` forces it.
fn wants_fullscreen() -> bool {
    match std::env::var("NFS_FULLSCREEN").ok().as_deref().map(str::trim) {
        Some("1") => true,
        Some("0") => false,
        _ => !matches!(video_driver().to_ascii_lowercase().as_str(), "x11" | "wayland" | "windows" | "cocoa"),
    }
}

pub(crate) fn display_name(display: i32) -> Option<String> {
    // SAFETY: NULL or a C string owned by SDL
    let p = unsafe { sys::SDL_GetDisplayName(display) };
    (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
}

pub(crate) fn display_modes(display: i32) -> Vec<VideoModeHandle> {
    let mut out = Vec::new();
    // SAFETY: modes queried by index into a plain struct
    unsafe {
        for i in 0..sys::SDL_GetNumDisplayModes(display).max(0) {
            let mut m: sys::SDL_DisplayMode = std::mem::zeroed();
            if sys::SDL_GetDisplayMode(display, i, &mut m) == 0 && m.w > 0 && m.h > 0 {
                out.push(VideoModeHandle {
                    size: PhysicalSize::new(m.w as u32, m.h as u32),
                    bits: (((m.format >> 8) & 0xFF) as u16).max(16),
                    refresh_mhz: m.refresh_rate.max(0) as u32 * 1000,
                });
            }
        }
    }
    out
}

/// One window's state, shared with the loop (redraw requests) and the renderer.
pub struct Shared {
    win: AtomicPtr<sys::SDL_Window>,
    id: AtomicU32,
    api: Mutex<GfxApi>,
    gl: Mutex<usize>,
    redraw: AtomicBool,
    fullscreen: bool,
    title: Mutex<String>,
    size: Mutex<PhysicalSize<u32>>,
    visible: bool,
}

// SAFETY: the SDL window is only used from the thread that owns the event loop / the renderer's
// core, which the game hands over as a whole (one thread at a time)
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

impl Drop for Shared {
    fn drop(&mut self) {
        let gl = *self.gl.get_mut().unwrap();
        // SAFETY: the context and the window were made by this object
        unsafe {
            if gl != 0 {
                sys::SDL_GL_DeleteContext(gl as sys::SDL_GLContext);
            }
            let w = *self.win.get_mut();
            if !w.is_null() {
                sys::SDL_DestroyWindow(w);
            }
        }
    }
}

/// winit's `Window`, over an `SDL_Window`.
pub struct Window {
    pub(crate) shared: Arc<Shared>,
}

impl std::fmt::Debug for Window {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Window").field("id", &self.shared.id.load(Ordering::Relaxed)).finish()
    }
}

fn make_window(title: &str, size: PhysicalSize<u32>, api: GfxApi, fullscreen: bool, visible: bool, resizable: bool) -> Result<*mut sys::SDL_Window, OsError> {
    use sys::SDL_WindowFlags as F;
    let mut flags = F::SDL_WINDOW_ALLOW_HIGHDPI as u32;
    flags |= match api {
        GfxApi::Gles => F::SDL_WINDOW_OPENGL as u32,
        GfxApi::Vulkan => F::SDL_WINDOW_VULKAN as u32,
    };
    if fullscreen {
        flags |= F::SDL_WINDOW_FULLSCREEN_DESKTOP as u32;
    } else if resizable {
        flags |= F::SDL_WINDOW_RESIZABLE as u32;
    }
    if !visible {
        flags |= F::SDL_WINDOW_HIDDEN as u32;
    }
    if api == GfxApi::Gles {
        use sys::SDL_GLattr as A;
        // SAFETY: attribute setters, before the window (EGL picks its config at window creation)
        unsafe {
            sys::SDL_GL_SetAttribute(A::SDL_GL_CONTEXT_PROFILE_MASK, sys::SDL_GLprofile::SDL_GL_CONTEXT_PROFILE_ES as i32);
            sys::SDL_GL_SetAttribute(A::SDL_GL_CONTEXT_MAJOR_VERSION, 3);
            sys::SDL_GL_SetAttribute(A::SDL_GL_CONTEXT_MINOR_VERSION, 0);
            sys::SDL_GL_SetAttribute(A::SDL_GL_RED_SIZE, 8);
            sys::SDL_GL_SetAttribute(A::SDL_GL_GREEN_SIZE, 8);
            sys::SDL_GL_SetAttribute(A::SDL_GL_BLUE_SIZE, 8);
            sys::SDL_GL_SetAttribute(A::SDL_GL_ALPHA_SIZE, 0);
            // wgpu draws into its own textures (with its own depth); the window only receives the blit
            sys::SDL_GL_SetAttribute(A::SDL_GL_DEPTH_SIZE, 0);
            sys::SDL_GL_SetAttribute(A::SDL_GL_STENCIL_SIZE, 0);
            sys::SDL_GL_SetAttribute(A::SDL_GL_DOUBLEBUFFER, 1);
        }
    }
    let t = CString::new(title.replace('\0', " ")).unwrap();
    const CENTERED: i32 = 0x2FFF0000;
    // SAFETY: valid title; SDL returns NULL on failure
    let w = unsafe { sys::SDL_CreateWindow(t.as_ptr(), CENTERED, CENTERED, size.width.max(1) as i32, size.height.max(1) as i32, flags) };
    if w.is_null() {
        return Err(OsError(sdl_error()));
    }
    Ok(w)
}

impl Window {
    pub fn default_attributes() -> WindowAttributes {
        WindowAttributes::default()
    }

    pub(crate) fn create(attrs: &WindowAttributes) -> Result<Window, OsError> {
        let fullscreen = wants_fullscreen();
        let size = attrs.inner_size.map(Size::to_physical).unwrap_or(PhysicalSize::new(640, 480));
        let mut api = GfxApi::from_env();
        let w = match make_window(&attrs.title, size, api, fullscreen, attrs.visible, attrs.resizable) {
            Ok(w) => w,
            // no Vulkan driver (SDL_Vulkan_LoadLibrary failed): the GLES window every firmware has
            Err(e) if api == GfxApi::Vulkan => {
                tracing::warn!(error = %e, "Vulkan window refused, using GLES");
                api = GfxApi::Gles;
                make_window(&attrs.title, size, api, fullscreen, attrs.visible, attrs.resizable)?
            }
            Err(e) => return Err(e),
        };
        if fullscreen && std::env::var("NFS_CURSOR").map_or(true, |v| v.trim() != "1") {
            // SAFETY: plain toggle
            unsafe { sys::SDL_ShowCursor(0) };
        }
        let shared = Arc::new(Shared {
            win: AtomicPtr::new(w),
            // SAFETY: a live window
            id: AtomicU32::new(unsafe { sys::SDL_GetWindowID(w) }),
            api: Mutex::new(api),
            gl: Mutex::new(0),
            redraw: AtomicBool::new(false),
            fullscreen,
            title: Mutex::new(attrs.title.clone()),
            size: Mutex::new(size),
            visible: attrs.visible,
        });
        let win = Window { shared };
        if let Some(icon) = attrs.window_icon.as_ref().filter(|_| !fullscreen) {
            win.set_icon(icon);
        }
        let s = win.inner_size();
        tracing::info!(api = ?api, fullscreen, width = s.width, height = s.height, "SDL2 window");
        Ok(win)
    }

    fn set_icon(&self, icon: &crate::window::Icon) {
        let mut px = icon.rgba.clone();
        // SAFETY: the surface borrows `px` only for the duration of the call
        unsafe {
            let s = sys::SDL_CreateRGBSurfaceFrom(px.as_mut_ptr() as *mut _, icon.width as i32, icon.height as i32, 32, icon.width as i32 * 4, 0x0000_00FF, 0x0000_FF00, 0x00FF_0000, 0xFF00_0000);
            if !s.is_null() {
                sys::SDL_SetWindowIcon(self.raw(), s);
                sys::SDL_FreeSurface(s);
            }
        }
    }

    fn raw(&self) -> *mut sys::SDL_Window {
        self.shared.win.load(Ordering::Acquire)
    }

    pub fn id(&self) -> WindowId {
        WindowId(self.shared.id.load(Ordering::Relaxed))
    }

    pub fn request_redraw(&self) {
        self.shared.redraw.store(true, Ordering::Release);
    }

    pub fn pre_present_notify(&self) {}

    /// The drawable size in pixels.
    pub fn inner_size(&self) -> PhysicalSize<u32> {
        let (mut w, mut h) = (0i32, 0i32);
        // SAFETY: a live window, out-params
        unsafe {
            match self.gfx_api() {
                GfxApi::Gles => sys::SDL_GL_GetDrawableSize(self.raw(), &mut w, &mut h),
                GfxApi::Vulkan => sys::SDL_Vulkan_GetDrawableSize(self.raw(), &mut w, &mut h),
            }
            if w <= 0 || h <= 0 {
                sys::SDL_GetWindowSize(self.raw(), &mut w, &mut h);
            }
        }
        PhysicalSize::new(w.max(1) as u32, h.max(1) as u32)
    }

    pub fn outer_size(&self) -> PhysicalSize<u32> {
        self.inner_size()
    }

    /// Pixels per window coordinate (1 on the handhelds).
    pub fn scale_factor(&self) -> f64 {
        let (mut w, mut h) = (0i32, 0i32);
        // SAFETY: a live window
        unsafe { sys::SDL_GetWindowSize(self.raw(), &mut w, &mut h) };
        if w <= 0 { 1.0 } else { self.inner_size().width as f64 / w as f64 }
    }

    pub fn set_title(&self, title: &str) {
        *self.shared.title.lock().unwrap() = title.to_string();
        if self.shared.fullscreen {
            return;
        }
        let t = CString::new(title.replace('\0', " ")).unwrap();
        // SAFETY: a live window, valid string
        unsafe { sys::SDL_SetWindowTitle(self.raw(), t.as_ptr()) };
    }

    pub fn set_visible(&self, _visible: bool) {}

    pub fn set_cursor_visible(&self, visible: bool) {
        // SAFETY: plain toggle
        unsafe { sys::SDL_ShowCursor(visible as i32) };
    }

    pub fn is_minimized(&self) -> Option<bool> {
        // SAFETY: a live window
        let f = unsafe { sys::SDL_GetWindowFlags(self.raw()) };
        // a fullscreen handheld window is always on screen (the Mali fbdev driver leaves it flagged
        // HIDDEN, which held races at the engine's 30 fps hidden-window cap)
        if self.shared.fullscreen {
            return Some(false);
        }
        Some(f & (sys::SDL_WindowFlags::SDL_WINDOW_MINIMIZED as u32 | sys::SDL_WindowFlags::SDL_WINDOW_HIDDEN as u32) != 0 && self.shared.visible)
    }

    pub fn has_focus(&self) -> bool {
        true
    }

    pub fn current_monitor(&self) -> Option<MonitorHandle> {
        // SAFETY: a live window
        let d = unsafe { sys::SDL_GetWindowDisplayIndex(self.raw()) };
        Some(MonitorHandle { display: d.max(0) })
    }

    pub fn primary_monitor(&self) -> Option<MonitorHandle> {
        Some(MonitorHandle { display: 0 })
    }

    /// A windowed desktop run resizes; a fullscreen handheld keeps the panel's mode (the game
    /// renders at the window's size whatever mode the menu picked).
    pub fn request_inner_size<S: Into<Size>>(&self, size: S) -> Option<PhysicalSize<u32>> {
        let s = size.into().to_physical();
        if self.shared.fullscreen {
            return Some(self.inner_size());
        }
        *self.shared.size.lock().unwrap() = s;
        // SAFETY: a live window
        unsafe { sys::SDL_SetWindowSize(self.raw(), s.width as i32, s.height as i32) };
        None
    }

    // ---- ReNFS5 additions: the graphics context -------------------------------------------

    /// The API this window was made for.
    pub fn gfx_api(&self) -> GfxApi {
        *self.shared.api.lock().unwrap()
    }

    /// The raw `SDL_Window` (for diagnostics).
    pub fn sdl_window(&self) -> *mut c_void {
        self.raw() as *mut c_void
    }

    /// Make the window again for `api` (the renderer's fallback from Vulkan to GLES when Vulkan
    /// opens but has no usable adapter). Only before anything was drawn.
    pub fn recreate_for(&self, api: GfxApi) -> Result<(), OsError> {
        let mut cur = self.shared.api.lock().unwrap();
        if *cur == api {
            return Ok(());
        }
        let mut gl = self.shared.gl.lock().unwrap();
        let title = self.shared.title.lock().unwrap().clone();
        let size = *self.shared.size.lock().unwrap();
        // SAFETY: the old context/window are ours and no longer used by the caller
        unsafe {
            if *gl != 0 {
                sys::SDL_GL_DeleteContext(*gl as sys::SDL_GLContext);
                *gl = 0;
            }
            sys::SDL_DestroyWindow(self.raw());
        }
        let w = make_window(&title, size, api, self.shared.fullscreen, self.shared.visible, true)?;
        self.shared.win.store(w, Ordering::Release);
        // SAFETY: a live window
        self.shared.id.store(unsafe { sys::SDL_GetWindowID(w) }, Ordering::Relaxed);
        *cur = api;
        tracing::info!(api = ?api, "SDL2 window made again");
        Ok(())
    }

    /// The window's GLES context, made (3.2, else 3.1, else 3.0) and current on the calling thread.
    pub fn gl_context(&self) -> Result<(), OsError> {
        use sys::SDL_GLattr as A;
        let mut gl = self.shared.gl.lock().unwrap();
        if *gl == 0 {
            for minor in [2, 1, 0] {
                // SAFETY: attribute setter, then context creation on a GL window
                let ctx = unsafe {
                    sys::SDL_GL_SetAttribute(A::SDL_GL_CONTEXT_MINOR_VERSION, minor);
                    sys::SDL_GL_CreateContext(self.raw())
                };
                if !ctx.is_null() {
                    *gl = ctx as usize;
                    tracing::info!(version = %format!("ES 3.{minor}"), "GL context");
                    break;
                }
                tracing::debug!(minor, error = %sdl_error(), "GL context refused");
            }
            if *gl == 0 {
                return Err(OsError(format!("no GLES 3 context: {}", sdl_error())));
            }
            let interval: i32 = std::env::var("NFS_VSYNC").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0);
            // SAFETY: the context is current (SDL_GL_CreateContext makes it so)
            unsafe { sys::SDL_GL_SetSwapInterval(interval) };
        }
        // SAFETY: our window and context
        if unsafe { sys::SDL_GL_MakeCurrent(self.raw(), *gl as sys::SDL_GLContext) } != 0 {
            return Err(OsError(sdl_error()));
        }
        Ok(())
    }

    /// Make the GL context current on this thread (`true`) or release it (`false`) — for handing the
    /// renderer to another thread.
    pub fn gl_make_current(&self, current: bool) {
        let gl = *self.shared.gl.lock().unwrap();
        if gl == 0 {
            return;
        }
        // SAFETY: our window and context
        unsafe { sys::SDL_GL_MakeCurrent(self.raw(), if current { gl as sys::SDL_GLContext } else { std::ptr::null_mut() }) };
    }

    /// A GL/GLES entry point of the current context.
    pub fn gl_get_proc_address(&self, name: &str) -> *const c_void {
        let n = CString::new(name).unwrap();
        // SAFETY: valid name
        unsafe { sys::SDL_GL_GetProcAddress(n.as_ptr()) as *const c_void }
    }

    /// Show the GL back buffer.
    pub fn gl_swap(&self) {
        // SAFETY: a GL window with its context current
        unsafe { sys::SDL_GL_SwapWindow(self.raw()) };
    }

    /// The Vulkan instance extensions SDL2 needs for this window's surface.
    pub fn vulkan_instance_extensions(&self) -> Result<Vec<CString>, OsError> {
        let mut n: u32 = 0;
        // SAFETY: two-call enumeration
        unsafe {
            if sys::SDL_Vulkan_GetInstanceExtensions(self.raw(), &mut n, std::ptr::null_mut()) == sys::SDL_bool::SDL_FALSE {
                return Err(OsError(sdl_error()));
            }
            let mut names: Vec<*const std::ffi::c_char> = vec![std::ptr::null(); n as usize];
            if sys::SDL_Vulkan_GetInstanceExtensions(self.raw(), &mut n, names.as_mut_ptr()) == sys::SDL_bool::SDL_FALSE {
                return Err(OsError(sdl_error()));
            }
            Ok(names.into_iter().take(n as usize).map(|p| CStr::from_ptr(p).to_owned()).collect())
        }
    }

    /// The window's `VkSurfaceKHR` on `instance` (a `VkInstance` handle).
    pub unsafe fn vulkan_create_surface(&self, instance: u64) -> Result<u64, OsError> {
        let mut surface: sys::VkSurfaceKHR = 0;
        // SAFETY: the caller's live instance, made with `vulkan_instance_extensions`
        if unsafe { sys::SDL_Vulkan_CreateSurface(self.raw(), instance as sys::VkInstance, &mut surface) } == sys::SDL_bool::SDL_FALSE {
            return Err(OsError(sdl_error()));
        }
        Ok(surface)
    }
}

/// The raw keyboard event (read through our own struct: SDL's scancode enum has gaps, and a
/// Rust enum must not hold a value it does not name).
#[repr(C)]
struct RawKey {
    type_: u32,
    timestamp: u32,
    window_id: u32,
    state: u8,
    repeat: u8,
    _p2: u8,
    _p3: u8,
    scancode: u32,
    sym: i32,
    mods: u16,
}

const SHIFT_MASK: u16 = 0x0001 | 0x0002;

fn take_events(out: &mut Vec<sys::SDL_Event>) {
    // SAFETY: SDL fills the events it returns
    unsafe {
        sys::SDL_PumpEvents();
        // everything but the joystick / controller events (0x600..0x6FF), which stay queued for nfs-input
        for (lo, hi) in [(0x100u32, 0x5FFu32), (0x700, 0xFFFF)] {
            loop {
                let mut ev: sys::SDL_Event = std::mem::zeroed();
                if sys::SDL_PeepEvents(&mut ev, 1, sys::SDL_eventaction::SDL_GETEVENT, lo, hi) <= 0 {
                    break;
                }
                out.push(ev);
            }
        }
    }
}

struct Pointer {
    scale_x: f64,
    scale_y: f64,
}

fn pointer_scale(win: &Window) -> Pointer {
    let (mut w, mut h) = (0i32, 0i32);
    // SAFETY: a live window
    unsafe { sys::SDL_GetWindowSize(win.raw(), &mut w, &mut h) };
    let s = win.inner_size();
    Pointer { scale_x: if w > 0 { s.width as f64 / w as f64 } else { 1.0 }, scale_y: if h > 0 { s.height as f64 / h as f64 } else { 1.0 } }
}

pub(crate) fn run<A: ApplicationHandler>(el: &ActiveEventLoop, app: &mut A) {
    app.new_events(el, crate::event::StartCause::Init);
    app.resumed(el);
    let mut events = Vec::with_capacity(64);
    let dev = DeviceId(0);
    let mut occluded = false;
    while !el.exit.get() {
        app.new_events(el, crate::event::StartCause::Poll);
        events.clear();
        take_events(&mut events);
        let windows: Vec<Arc<Shared>> = {
            let mut ws = el.windows.borrow_mut();
            ws.retain(|w| w.strong_count() > 0);
            ws.iter().filter_map(|w| w.upgrade()).collect()
        };
        let main = windows.first().map(|s| Window { shared: s.clone() });
        let wid = main.as_ref().map_or(WindowId(0), |w| w.id());
        let emit = |app: &mut A, e: WindowEvent| app.window_event(el, wid, e);
        for ev in &events {
            // SAFETY: each arm reads the union member its type names
            let ty = unsafe { ev.type_ };
            use sys::SDL_EventType as T;
            match ty {
                t if t == T::SDL_QUIT as u32 => emit(app, WindowEvent::CloseRequested),
                t if t == T::SDL_WINDOWEVENT as u32 => {
                    let w = unsafe { ev.window };
                    use sys::SDL_WindowEventID as W;
                    match w.event as u32 {
                        e if e == W::SDL_WINDOWEVENT_CLOSE as u32 => emit(app, WindowEvent::CloseRequested),
                        e if e == W::SDL_WINDOWEVENT_SIZE_CHANGED as u32 => {
                            if let Some(m) = main.as_ref() {
                                emit(app, WindowEvent::Resized(m.inner_size()));
                                m.request_redraw();
                            }
                        }
                        e if e == W::SDL_WINDOWEVENT_FOCUS_GAINED as u32 => emit(app, WindowEvent::Focused(true)),
                        e if e == W::SDL_WINDOWEVENT_FOCUS_LOST as u32 => emit(app, WindowEvent::Focused(false)),
                        e if e == W::SDL_WINDOWEVENT_LEAVE as u32 => emit(app, WindowEvent::CursorLeft { device_id: dev }),
                        e if e == W::SDL_WINDOWEVENT_ENTER as u32 => emit(app, WindowEvent::CursorEntered { device_id: dev }),
                        e if e == W::SDL_WINDOWEVENT_HIDDEN as u32 || e == W::SDL_WINDOWEVENT_MINIMIZED as u32 => {
                            if !occluded {
                                occluded = true;
                                emit(app, WindowEvent::Occluded(true));
                            }
                        }
                        e if e == W::SDL_WINDOWEVENT_SHOWN as u32 || e == W::SDL_WINDOWEVENT_RESTORED as u32 || e == W::SDL_WINDOWEVENT_EXPOSED as u32 => {
                            if occluded {
                                occluded = false;
                                emit(app, WindowEvent::Occluded(false));
                            }
                            if let Some(m) = main.as_ref() {
                                m.request_redraw();
                            }
                        }
                        _ => {}
                    }
                }
                t if t == T::SDL_KEYDOWN as u32 || t == T::SDL_KEYUP as u32 => {
                    // SAFETY: SDL_KeyboardEvent's layout (RawKey mirrors it)
                    let k = unsafe { &*(ev as *const sys::SDL_Event as *const RawKey) };
                    let shift = k.mods & SHIFT_MASK != 0;
                    let (logical_key, text) = crate::keys::logical(k.scancode, k.sym, shift);
                    let pressed = t == T::SDL_KEYDOWN as u32;
                    let event = KeyEvent {
                        physical_key: crate::keys::physical(k.scancode),
                        logical_key,
                        text: if pressed { text } else { None },
                        location: crate::keys::location(k.scancode),
                        state: if pressed { ElementState::Pressed } else { ElementState::Released },
                        repeat: k.repeat != 0,
                    };
                    emit(app, WindowEvent::KeyboardInput { device_id: dev, event, is_synthetic: false });
                }
                t if t == T::SDL_MOUSEMOTION as u32 => {
                    let m = unsafe { ev.motion };
                    let p = main.as_ref().map_or(Pointer { scale_x: 1.0, scale_y: 1.0 }, pointer_scale);
                    app.device_event(el, dev, DeviceEvent::MouseMotion { delta: (m.xrel as f64, m.yrel as f64) });
                    emit(app, WindowEvent::CursorMoved { device_id: dev, position: PhysicalPosition::new(m.x as f64 * p.scale_x, m.y as f64 * p.scale_y) });
                }
                t if t == T::SDL_MOUSEBUTTONDOWN as u32 || t == T::SDL_MOUSEBUTTONUP as u32 => {
                    let b = unsafe { ev.button };
                    let button = match b.button {
                        1 => MouseButton::Left,
                        2 => MouseButton::Middle,
                        3 => MouseButton::Right,
                        4 => MouseButton::Back,
                        5 => MouseButton::Forward,
                        n => MouseButton::Other(n as u16),
                    };
                    let state = if t == T::SDL_MOUSEBUTTONDOWN as u32 { ElementState::Pressed } else { ElementState::Released };
                    emit(app, WindowEvent::MouseInput { device_id: dev, state, button });
                }
                t if t == T::SDL_MOUSEWHEEL as u32 => {
                    let w = unsafe { ev.wheel };
                    emit(app, WindowEvent::MouseWheel { device_id: dev, delta: MouseScrollDelta::LineDelta(w.x as f32, w.y as f32), phase: TouchPhase::Moved });
                }
                _ => {}
            }
            if el.exit.get() {
                break;
            }
        }
        if el.exit.get() {
            break;
        }
        let mut drew = false;
        for s in &windows {
            if s.redraw.swap(false, Ordering::AcqRel) {
                drew = true;
                app.window_event(el, WindowId(s.id.load(Ordering::Relaxed)), WindowEvent::RedrawRequested);
            }
        }
        app.about_to_wait(el);
        if !drew && events.is_empty() {
            // nothing to draw: wait a little for the next event instead of spinning
            let ms = match el.control_flow.get() {
                ControlFlow::Poll => 1,
                ControlFlow::Wait => 10,
                ControlFlow::WaitUntil(t) => t.saturating_duration_since(std::time::Instant::now()).as_millis().min(10) as i32,
            };
            // SAFETY: NULL event = only wait
            unsafe { sys::SDL_WaitEventTimeout(std::ptr::null_mut(), ms) };
        }
    }
    app.exiting(el);
}
