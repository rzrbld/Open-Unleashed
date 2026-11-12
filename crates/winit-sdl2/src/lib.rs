//! ReNFS5: the part of the winit 0.30 API the game uses, implemented over the firmware's SDL2.
//!
//! Handheld firmwares (muOS, Knulli, ROCKNIX, …) have no X11 or Wayland: games draw through
//! SDL2's KMSDRM (or vendor) video driver, read the pad through SDL2, and PortMaster's controls
//! (`gptokeyb`, `SDL_GAMECONTROLLERCONFIG`) are made for SDL2. This crate is linked under the
//! name `winit`, so the game keeps its window / event code unchanged.
//!
//! What it adds beyond winit lives in [`sdl`]: the graphics API the window was made for (GLES,
//! the default, or Vulkan), the GL context and its swap, and the Vulkan surface — the renderer
//! uses those instead of a raw window handle.
//!
//! SDL2's event queue is shared with the controller code (`nfs-input`): this layer takes every
//! event **except** the joystick / game controller ones (`SDL_JOYAXISMOTION`..`SDL_CONTROLLER*`,
//! 0x600..0x6FF), which stay queued for the controller code to take with `SDL_PeepEvents`.
//!
//! `keyboard.rs` is winit's own (Apache-2.0, see `LICENSE-winit`).

#![allow(clippy::missing_safety_doc)]

pub mod keyboard;

mod keys;
pub mod sdl;

pub mod dpi {
    //! Sizes and positions, as winit's `dpi` names them (no DPI scaling on the handhelds).
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
    pub struct PhysicalSize<P> {
        pub width: P,
        pub height: P,
    }
    impl<P> PhysicalSize<P> {
        pub const fn new(width: P, height: P) -> Self {
            Self { width, height }
        }
    }
    #[derive(Debug, Copy, Clone, PartialEq, Default)]
    pub struct LogicalSize<P> {
        pub width: P,
        pub height: P,
    }
    impl<P> LogicalSize<P> {
        pub const fn new(width: P, height: P) -> Self {
            Self { width, height }
        }
    }
    #[derive(Debug, Copy, Clone, PartialEq, Default)]
    pub struct PhysicalPosition<P> {
        pub x: P,
        pub y: P,
    }
    impl<P> PhysicalPosition<P> {
        pub const fn new(x: P, y: P) -> Self {
            Self { x, y }
        }
    }
    #[derive(Debug, Copy, Clone, PartialEq, Default)]
    pub struct LogicalPosition<P> {
        pub x: P,
        pub y: P,
    }
    /// A size in either unit (scale factor 1: both are pixels).
    #[derive(Debug, Copy, Clone, PartialEq)]
    pub enum Size {
        Physical(PhysicalSize<u32>),
        Logical(LogicalSize<f64>),
    }
    impl Size {
        pub fn to_physical(self) -> PhysicalSize<u32> {
            match self {
                Size::Physical(s) => s,
                Size::Logical(s) => PhysicalSize::new(s.width.round().max(1.0) as u32, s.height.round().max(1.0) as u32),
            }
        }
    }
    macro_rules! size_from {
        ($($t:ty),*) => {$(
            impl From<PhysicalSize<$t>> for Size {
                fn from(s: PhysicalSize<$t>) -> Self { Size::Physical(PhysicalSize::new(s.width as u32, s.height as u32)) }
            }
            impl From<LogicalSize<$t>> for Size {
                fn from(s: LogicalSize<$t>) -> Self { Size::Logical(LogicalSize::new(s.width as f64, s.height as f64)) }
            }
        )*};
    }
    size_from!(u32, i32, f32, f64);
}

pub mod error {
    use std::fmt;
    #[derive(Debug)]
    pub struct OsError(pub String);
    impl fmt::Display for OsError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "SDL2: {}", self.0)
        }
    }
    impl std::error::Error for OsError {}
    #[derive(Debug)]
    pub enum EventLoopError {
        Os(OsError),
        ExitFailure(i32),
    }
    impl fmt::Display for EventLoopError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                EventLoopError::Os(e) => e.fmt(f),
                EventLoopError::ExitFailure(c) => write!(f, "exit failure {c}"),
            }
        }
    }
    impl std::error::Error for EventLoopError {}
    #[derive(Debug)]
    pub struct BadIcon(pub String);
    impl fmt::Display for BadIcon {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "bad icon: {}", self.0)
        }
    }
    impl std::error::Error for BadIcon {}
}

pub mod event {
    //! The events the game reads (winit's names and fields).
    use crate::dpi::{PhysicalPosition, PhysicalSize};
    use crate::keyboard::{Key, KeyLocation, PhysicalKey, SmolStr};

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct DeviceId(pub(crate) u32);
    impl DeviceId {
        pub const fn dummy() -> Self {
            DeviceId(0)
        }
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub enum ElementState {
        Pressed,
        Released,
    }
    impl ElementState {
        pub fn is_pressed(self) -> bool {
            self == ElementState::Pressed
        }
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub enum MouseButton {
        Left,
        Right,
        Middle,
        Back,
        Forward,
        Other(u16),
    }

    #[derive(Debug, Copy, Clone, PartialEq)]
    pub enum MouseScrollDelta {
        LineDelta(f32, f32),
        PixelDelta(PhysicalPosition<f64>),
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub enum TouchPhase {
        Started,
        Moved,
        Ended,
        Cancelled,
    }

    /// winit's `KeyEvent`, without its private platform part (so it can be built here).
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct KeyEvent {
        pub physical_key: PhysicalKey,
        pub logical_key: Key,
        pub text: Option<SmolStr>,
        pub location: KeyLocation,
        pub state: ElementState,
        pub repeat: bool,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum WindowEvent {
        Resized(PhysicalSize<u32>),
        CloseRequested,
        Destroyed,
        Focused(bool),
        Occluded(bool),
        KeyboardInput { device_id: DeviceId, event: KeyEvent, is_synthetic: bool },
        CursorMoved { device_id: DeviceId, position: PhysicalPosition<f64> },
        CursorEntered { device_id: DeviceId },
        CursorLeft { device_id: DeviceId },
        MouseWheel { device_id: DeviceId, delta: MouseScrollDelta, phase: TouchPhase },
        MouseInput { device_id: DeviceId, state: ElementState, button: MouseButton },
        ScaleFactorChanged { scale_factor: f64 },
        RedrawRequested,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum DeviceEvent {
        MouseMotion { delta: (f64, f64) },
        Added,
        Removed,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum StartCause {
        Poll,
        Init,
    }
}

pub mod monitor {
    use crate::dpi::PhysicalSize;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct MonitorHandle {
        pub(crate) display: i32,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct VideoModeHandle {
        pub(crate) size: PhysicalSize<u32>,
        pub(crate) bits: u16,
        pub(crate) refresh_mhz: u32,
    }
    impl VideoModeHandle {
        pub fn size(&self) -> PhysicalSize<u32> {
            self.size
        }
        pub fn bit_depth(&self) -> u16 {
            self.bits
        }
        pub fn refresh_rate_millihertz(&self) -> u32 {
            self.refresh_mhz
        }
    }

    impl MonitorHandle {
        pub fn name(&self) -> Option<String> {
            crate::sdl::display_name(self.display)
        }
        /// The display's modes as SDL2 lists them. A fullscreen handheld window keeps the panel's
        /// mode whatever is picked (the game renders at the window's size).
        pub fn video_modes(&self) -> impl Iterator<Item = VideoModeHandle> + use<> {
            crate::sdl::display_modes(self.display).into_iter()
        }
        pub fn size(&self) -> PhysicalSize<u32> {
            crate::sdl::display_modes(self.display).first().map(|m| m.size).unwrap_or_default()
        }
        pub fn scale_factor(&self) -> f64 {
            1.0
        }
    }
}

pub mod window {
    pub use crate::sdl::Window;
    use crate::dpi::Size;
    use crate::error::BadIcon;

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct WindowId(pub(crate) u32);
    impl WindowId {
        pub const fn dummy() -> Self {
            WindowId(0)
        }
    }
    impl From<WindowId> for u64 {
        fn from(w: WindowId) -> u64 {
            w.0 as u64
        }
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
    pub enum WindowLevel {
        AlwaysOnBottom,
        #[default]
        Normal,
        AlwaysOnTop,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Icon {
        pub(crate) rgba: Vec<u8>,
        pub(crate) width: u32,
        pub(crate) height: u32,
    }
    impl Icon {
        pub fn from_rgba(rgba: Vec<u8>, width: u32, height: u32) -> Result<Self, BadIcon> {
            if rgba.len() as u64 != width as u64 * height as u64 * 4 {
                return Err(BadIcon(format!("{} bytes for {width}x{height}", rgba.len())));
            }
            Ok(Icon { rgba, width, height })
        }
    }

    #[derive(Debug, Clone)]
    pub struct WindowAttributes {
        pub title: String,
        pub inner_size: Option<Size>,
        pub visible: bool,
        pub active: bool,
        pub window_level: WindowLevel,
        pub window_icon: Option<Icon>,
        pub resizable: bool,
    }
    impl Default for WindowAttributes {
        fn default() -> Self {
            Self { title: "winit window".into(), inner_size: None, visible: true, active: true, window_level: WindowLevel::Normal, window_icon: None, resizable: true }
        }
    }
    impl WindowAttributes {
        pub fn with_title(mut self, t: impl Into<String>) -> Self {
            self.title = t.into();
            self
        }
        pub fn with_inner_size<S: Into<Size>>(mut self, s: S) -> Self {
            self.inner_size = Some(s.into());
            self
        }
        pub fn with_visible(mut self, v: bool) -> Self {
            self.visible = v;
            self
        }
        pub fn with_active(mut self, a: bool) -> Self {
            self.active = a;
            self
        }
        pub fn with_window_level(mut self, l: WindowLevel) -> Self {
            self.window_level = l;
            self
        }
        pub fn with_window_icon(mut self, i: Option<Icon>) -> Self {
            self.window_icon = i;
            self
        }
        pub fn with_resizable(mut self, r: bool) -> Self {
            self.resizable = r;
            self
        }
    }
}

pub mod application {
    use crate::event::{DeviceEvent, DeviceId, StartCause, WindowEvent};
    use crate::event_loop::ActiveEventLoop;
    use crate::window::WindowId;

    /// winit's `ApplicationHandler` (no user events).
    pub trait ApplicationHandler {
        fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) {
            let _ = (event_loop, cause);
        }
        fn resumed(&mut self, event_loop: &ActiveEventLoop);
        fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent);
        fn device_event(&mut self, event_loop: &ActiveEventLoop, device_id: DeviceId, event: DeviceEvent) {
            let _ = (event_loop, device_id, event);
        }
        fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
            let _ = event_loop;
        }
        fn suspended(&mut self, event_loop: &ActiveEventLoop) {
            let _ = event_loop;
        }
        fn exiting(&mut self, event_loop: &ActiveEventLoop) {
            let _ = event_loop;
        }
        fn memory_warning(&mut self, event_loop: &ActiveEventLoop) {
            let _ = event_loop;
        }
    }
}

pub mod event_loop {
    use crate::application::ApplicationHandler;
    use crate::error::{EventLoopError, OsError};
    use crate::window::{Window, WindowAttributes};
    use std::cell::{Cell, RefCell};
    use std::time::Instant;

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
    pub enum ControlFlow {
        #[default]
        Poll,
        Wait,
        WaitUntil(Instant),
    }

    pub struct EventLoopBuilder {
        _p: (),
    }
    impl EventLoopBuilder {
        pub fn build(&mut self) -> Result<EventLoop, EventLoopError> {
            crate::sdl::init_video().map_err(EventLoopError::Os)?;
            Ok(EventLoop { active: ActiveEventLoop::new() })
        }
    }

    pub struct EventLoop {
        active: ActiveEventLoop,
    }
    impl EventLoop {
        pub fn builder() -> EventLoopBuilder {
            EventLoopBuilder { _p: () }
        }
        pub fn new() -> Result<EventLoop, EventLoopError> {
            Self::builder().build()
        }
        pub fn set_control_flow(&self, cf: ControlFlow) {
            self.active.control_flow.set(cf);
        }
        pub fn run_app<A: ApplicationHandler>(self, app: &mut A) -> Result<(), EventLoopError> {
            crate::sdl::run(&self.active, app);
            Ok(())
        }
    }

    /// The loop as the handler sees it: exit, window creation.
    pub struct ActiveEventLoop {
        pub(crate) control_flow: Cell<ControlFlow>,
        pub(crate) exit: Cell<bool>,
        pub(crate) windows: RefCell<Vec<std::sync::Weak<crate::sdl::Shared>>>,
    }
    impl ActiveEventLoop {
        fn new() -> Self {
            Self { control_flow: Cell::new(ControlFlow::Poll), exit: Cell::new(false), windows: RefCell::new(Vec::new()) }
        }
        pub fn create_window(&self, attrs: WindowAttributes) -> Result<Window, OsError> {
            let w = Window::create(&attrs)?;
            self.windows.borrow_mut().push(std::sync::Arc::downgrade(&w.shared));
            Ok(w)
        }
        pub fn exit(&self) {
            self.exit.set(true);
        }
        pub fn exiting(&self) -> bool {
            self.exit.get()
        }
        pub fn set_control_flow(&self, cf: ControlFlow) {
            self.control_flow.set(cf);
        }
        pub fn control_flow(&self) -> ControlFlow {
            self.control_flow.get()
        }
        pub fn primary_monitor(&self) -> Option<crate::monitor::MonitorHandle> {
            Some(crate::monitor::MonitorHandle { display: 0 })
        }
    }
}

pub mod platform {
    //! winit's platform extension modules: none apply here.
}
