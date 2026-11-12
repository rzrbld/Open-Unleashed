//! The output: the offline dump, the device stream glue (SDL2 or cpal, `NFS_AUDIO`) and the public
//! `AudioOutput` API.

use super::*;

/// TEST GLUE (`NFS_AUDIO_DUMP=<file>`): no device. The mixer runs on the calling thread: every
/// call that changes something is applied at once, and every 4th renders 441 frames at 44100 Hz
/// stereo, appended as f32 LE to the file — a deterministic function of the game's calls, to
/// compare the mix of one build against another (`RE/notes/PERF.md`, F5).
pub(super) struct Offline {
    pub(super) core: Core,
    pub(super) n: u32,
    /// TEST GLUE (`NFS_AUDIO_DUMP_TICKS=1`): the mix is advanced by the game's ticks instead
    /// ([`AudioOutput::offline_game_tick`]): the game ticks seen so far and the mix-rate frames written
    pub(super) ticks: u64,
    pub(super) written: u64,
}

pub(super) static DUMP: std::sync::OnceLock<Mutex<std::fs::File>> = std::sync::OnceLock::new();

/// TEST GLUE: `NFS_AUDIO_DUMP_TICKS=1` (with `NFS_AUDIO_DUMP`) — the dump is the mix at its own rate
/// `[0x6B48DA]`, 16-bit stereo, `rate / 64` frames per game tick (`RE/notes/REFERENCE-2026-10-24-PA-AUDIOMIX.md`):
/// the same samples the original's driver pass hands DirectSound (0x58FBC0), on the game's clock.
pub(super) fn dump_ticked() -> bool {
    std::env::var_os("NFS_AUDIO_DUMP_TICKS").is_some_and(|v| !v.is_empty() && v != "0")
}

impl Offline {
    /// TEST GLUE: one game tick of mix ([`dump_ticked`]).
    pub(super) fn game_tick(&mut self) {
        self.core.drain();
        self.ticks += 1;
        if self.ticks == 1 {
            tracing::info!(game_tick = crate::trace_tick(), "audio dump: the first tick of mix (NFS_AUDIO_DUMP_TICKS)");
        }
        let rate = self.core.shared.mix_rate() as u64;
        let target = self.ticks * rate / 64;
        let n = target.saturating_sub(self.written) as usize;
        self.written = target;
        let mut buf = Vec::new();
        self.core.shared.render_into(&mut buf, n, rate as u32);
        let bytes: Vec<u8> = buf.iter().flat_map(|f| f.iter().flat_map(|x| ((x * 32768.0).round().clamp(-32768.0, 32767.0) as i16).to_le_bytes())).collect();
        use std::io::Write;
        if let Some(dump) = DUMP.get() {
            dump.lock().unwrap().write_all(&bytes).unwrap();
        }
        self.core.publish();
    }

    pub(super) fn step(&mut self) {
        self.core.drain();
        self.n += 1;
        if dump_ticked() {
            self.core.publish();
            return;
        }
        if self.n.is_multiple_of(4) {
            let mut buf = vec![0.0f32; 441 * 2];
            self.core.fill(&mut buf, 2, 44100.0);
            let bytes: Vec<u8> = buf.iter().flat_map(|x| x.to_le_bytes()).collect();
            use std::io::Write;
            // TEST GLUE: a muted run (`AudioOutput::new`, no `NFS_AUDIO_DUMP`) mixes the same way and writes nothing
            if let Some(dump) = DUMP.get() {
                dump.lock().unwrap().write_all(&bytes).unwrap();
            }
        }
        self.core.publish();
    }
}

/// The device buffer the stream asks for, in frames: `NFS_AUDIO_BUFFER` (0 = the device's own
/// default), else 256 on macOS (CoreAudio's default is 512), clamped to what the device supports;
/// other systems keep their default buffer.
// PORT-DEVIATION D2 (RE/notes/PORT-DEVIATIONS.md): a fixed 256-frame device buffer on macOS; the original's sound layer opens at 0x4A66B0 with 0x565720([0x655D44], 0x40000, 3).
pub(super) fn device_buffer_frames(supported: &cpal::SupportedStreamConfig) -> Option<u32> {
    let asked = std::env::var("NFS_AUDIO_BUFFER").ok().and_then(|v| v.parse::<u32>().ok());
    let n = asked.unwrap_or(if cfg!(target_os = "macos") { 256 } else { 0 });
    match (n, supported.buffer_size()) {
        (0, _) => None,
        (n, cpal::SupportedBufferSize::Range { min, max }) => Some(n.clamp(*min, *max)),
        (_, cpal::SupportedBufferSize::Unknown) => None,
    }
}

/// One voice start ([`AudioOutput::play`]): the arguments of the `play_word*` family in one struct.
/// [`PlayArgs::new`] is `play_word`; `.sel()` / `.untabled()` give the other three.
pub struct PlayArgs {
    pub clip: Arc<Clip>,
    pub volume: f32,
    pub word: u16,
    pub bend: u8,
    pub pan: u16,
    pub looping: bool,
    /// the request's byte `+6`
    pub sel: u8,
    /// written to the pause's voice table (`Voice_play`) or not (a bare `Voice_start`)
    pub tabled: bool,
    /// the request's word `+0x12`, the strip hook's value at the start ([`PlayArgs::filter`])
    pub filter: Option<i32>,
}

impl PlayArgs {
    /// `play_word`'s request: `sel` 0x7F, tabled.
    pub fn new(clip: Arc<Clip>, volume: f32, word: u16, bend: u8, pan: u16, looping: bool) -> Self {
        Self { clip, volume, word, bend, pan, looping, sel: 0x7F, tabled: true, filter: None }
    }

    /// **`Bank_playOneShot` 0x4A98E0's filter word**: the request's `+0x12` (0xFFFF from 0x561960)
    /// becomes **0xBB8** at 0x4A9B2B when the bank is not the player's engine bank (`[0x6564A0]` /
    /// `[0x6564B0]`, 0x4A9AE0…0x4A9AEA), `AUDIOQUALITY` is below 2 (0x4A9AEC) and the view is the
    /// cockpit (`[0x627F43] & 8`, the view word neither 0 nor 0xF, 0x443EE0 not exterior,
    /// 0x4A9AF5…0x4A9B29) — the same gate as `Voice_play`'s update ([`voice_play_filter`]). The start
    /// hands it on (0x561508 → 0x5897C0 → 0x589FE0, which calls the strip hook `+0xE0` with it,
    /// 0x596AC0), so such a voice is low-passed at 3000 Hz **from its first sample**, a one-shot
    /// included. `value` is [`voice_play_filter`]'s (0xFFFF or nothing: no filter). Measured on the
    /// running original (`REFERENCE-2026-10-24-PA-AUDIOMIX.md`: the wrong-way cue, 0xBB8 in its
    /// start request, 12.8 dB under the port's unfiltered one). *(Until 2026-10-24 the port read
    /// the start as returning before the filter — true of 0x4A9C10's update tail only — and played
    /// every one-shot in the cockpit unfiltered, a loop for its first update.)*
    pub fn filter(mut self, value: Option<i32>) -> Self {
        self.filter = value;
        self
    }

    /// With the request's byte `+6`.
    pub fn sel(mut self, sel: u8) -> Self {
        self.sel = sel;
        self
    }

    /// Not in the pause's voice table.
    pub fn untabled(mut self) -> Self {
        self.tabled = false;
        self
    }
}

/// The audio engine a device stream runs on (`NFS_AUDIO`, [`Backend::from_env`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// SDL2's audio subsystem — the firmware's own driver (ALSA, PulseAudio, PipeWire, …,
    /// `SDL_AUDIODRIVER`), what PortMaster ports rely on
    Sdl,
    /// cpal, which on Linux talks to ALSA directly
    Cpal,
}

impl Backend {
    /// PORT-DEVIATION (not in the original): `NFS_AUDIO` = `sdl` | `cpal` | `auto` (the default,
    /// SDL2 first). The backends tried in order: the asked one, then the other one as a fallback.
    pub fn order_from_env() -> [Backend; 2] {
        let v = std::env::var("NFS_AUDIO").unwrap_or_default().to_ascii_lowercase();
        match v.trim() {
            "cpal" | "alsa" => [Backend::Cpal, Backend::Sdl],
            "" | "auto" | "sdl" | "sdl2" => [Backend::Sdl, Backend::Cpal],
            other => {
                tracing::warn!(value = other, "audio: unknown NFS_AUDIO (sdl | cpal | auto), using auto");
                [Backend::Sdl, Backend::Cpal]
            }
        }
    }
}

/// The open device stream of an [`AudioOutput`]; dropping it stops the device callback.
pub(super) enum Stream {
    Cpal(#[allow(dead_code)] cpal::Stream),
    Sdl(#[allow(dead_code)] SdlStream),
}

/// What the SDL2 callback works on, behind the `userdata` pointer. `core` is put in after the
/// device is open (its rate is the obtained one) and before it is unpaused: SDL never calls the
/// callback of a paused device, and `SDL_PauseAudioDevice` takes the device's lock, which orders
/// the write before the audio thread's first read.
struct SdlCtx {
    core: Option<Core>,
    channels: usize,
    rate: f64,
    /// the device buffer in frames (`have.samples`): SDL2 gives no playback timestamp, so this is
    /// the latency the callback reports
    frames: u64,
}

/// An open SDL2 playback device and its callback's state ([`SdlCtx`]). The audio subsystem is
/// initialised with `SDL_InitSubSystem` (ref-counted; the window side owns `SDL_Init`/`SDL_Quit`)
/// and quit again on drop.
pub(super) struct SdlStream {
    dev: sdl2_sys::SDL_AudioDeviceID,
    ctx: *mut SdlCtx,
}

// SAFETY: the context is only touched by SDL's audio thread while the device is open; this side
// only closes the device (which joins that thread) and then frees it. `Core` is `Send` (cpal's
// callback owns one on its own thread the same way).
unsafe impl Send for SdlStream {}

/// SDL's subsystem init / quit and device open / close are not thread-safe: two outputs opened
/// at once (the tests do) crash inside SDL. Every such call of this module holds this lock; the
/// audio callback never takes it.
static SDL_LOCK: Mutex<()> = Mutex::new(());

impl Drop for SdlStream {
    fn drop(&mut self) {
        let _g = SDL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            sdl2_sys::SDL_CloseAudioDevice(self.dev);
            drop(Box::from_raw(self.ctx));
            sdl2_sys::SDL_QuitSubSystem(sdl2_sys::SDL_INIT_AUDIO);
        }
    }
}

unsafe extern "C" fn sdl_callback(userdata: *mut std::ffi::c_void, stream: *mut u8, len: std::ffi::c_int) {
    // SAFETY: `userdata` is the `SdlCtx` box, alive until the device is closed; SDL hands a buffer
    // of `len` bytes in the format of the spec (f32, the format change is never allowed), aligned
    // for it (SDL allocates it with SDL_malloc / SIMD alignment).
    let ctx = unsafe { &mut *(userdata as *mut SdlCtx) };
    let n = len.max(0) as usize / std::mem::size_of::<f32>();
    let out = unsafe { std::slice::from_raw_parts_mut(stream as *mut f32, n) };
    let Some(core) = ctx.core.as_mut() else {
        out.fill(0.0);
        return;
    };
    core.drain();
    core.lat.frames.store(ctx.frames.max((n / ctx.channels.max(1)) as u64), Ordering::Relaxed);
    core.lat.at.store(core.lat.epoch.elapsed().as_nanos() as u64 + 1, Ordering::Release);
    core.fill(out, ctx.channels, ctx.rate);
    core.publish();
}

fn sdl_error() -> String {
    unsafe {
        let e = sdl2_sys::SDL_GetError();
        if e.is_null() { String::new() } else { std::ffi::CStr::from_ptr(e).to_string_lossy().into_owned() }
    }
}

/// The SDL2 device buffer asked for, in frames: `NFS_AUDIO_FRAMES` (a power of two is best;
/// clamped to 64..=8192), else 512.
fn sdl_buffer_frames() -> u16 {
    std::env::var("NFS_AUDIO_FRAMES").ok().and_then(|v| v.trim().parse::<u32>().ok()).unwrap_or(512).clamp(64, 8192) as u16
}

impl SdlStream {
    /// Opens the default playback device (f32, 44100 Hz stereo asked; SDL may change the rate and
    /// the channel count, never the format — it converts for us), with a mixer at the obtained
    /// rate, and starts it.
    fn open() -> anyhow::Result<(Self, Arc<Tx>, u32)> {
        use sdl2_sys as sdl;
        let _g = SDL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            if sdl::SDL_InitSubSystem(sdl::SDL_INIT_AUDIO) != 0 {
                anyhow::bail!("SDL_InitSubSystem(AUDIO): {}", sdl_error());
            }
        }
        let ctx = Box::into_raw(Box::new(SdlCtx { core: None, channels: 2, rate: 44100.0, frames: 0 }));
        let fail = |msg: String| -> anyhow::Error {
            unsafe {
                drop(Box::from_raw(ctx));
                sdl::SDL_QuitSubSystem(sdl::SDL_INIT_AUDIO);
            }
            anyhow::anyhow!(msg)
        };
        let want = sdl::SDL_AudioSpec {
            freq: 44100,
            format: sdl::AUDIO_F32SYS as sdl::SDL_AudioFormat,
            channels: 2,
            silence: 0,
            samples: sdl_buffer_frames(),
            padding: 0,
            size: 0,
            callback: Some(sdl_callback),
            userdata: ctx as *mut std::ffi::c_void,
        };
        let open = |flags: u32| -> (sdl::SDL_AudioDeviceID, sdl::SDL_AudioSpec) {
            let mut have: sdl::SDL_AudioSpec = unsafe { std::mem::zeroed() };
            let dev = unsafe { sdl::SDL_OpenAudioDevice(std::ptr::null(), 0, &want, &mut have, flags as std::ffi::c_int) };
            (dev, have)
        };
        let (mut dev, mut have) = open(sdl::SDL_AUDIO_ALLOW_FREQUENCY_CHANGE | sdl::SDL_AUDIO_ALLOW_CHANNELS_CHANGE);
        if dev != 0 && (have.format as u32 != sdl::AUDIO_F32SYS || have.channels == 0 || have.freq <= 0) {
            // not expected without SDL_AUDIO_ALLOW_FORMAT_CHANGE; SDL converts with no change allowed
            tracing::warn!(format = have.format, channels = have.channels, freq = have.freq, "audio (SDL2): unusable spec, reopening with no change allowed");
            unsafe { sdl::SDL_CloseAudioDevice(dev) };
            (dev, have) = open(0);
        }
        if dev == 0 {
            return Err(fail(format!("SDL_OpenAudioDevice: {}", sdl_error())));
        }
        let rate = have.freq as u32;
        let (tx, core) = Tx::new(rate, false);
        // SAFETY: the device is paused (SDL opens it so), its callback does not run yet
        unsafe {
            let c = &mut *ctx;
            c.core = Some(core.expect("a device mixer has its callback half"));
            c.channels = have.channels as usize;
            c.rate = rate as f64;
            c.frames = have.samples as u64;
            sdl::SDL_PauseAudioDevice(dev, 0);
        }
        let driver = unsafe {
            let d = sdl::SDL_GetCurrentAudioDriver();
            if d.is_null() { String::from("?") } else { std::ffi::CStr::from_ptr(d).to_string_lossy().into_owned() }
        };
        tracing::info!(
            backend = "sdl2",
            driver,
            rate,
            channels = have.channels,
            buffer_frames = have.samples,
            format = format_args!("{:#06x}", have.format),
            "audio output started"
        );
        Ok((SdlStream { dev, ctx }, tx, rate))
    }
}

pub struct AudioOutput {
    pub(super) _stream: Option<Stream>,
    pub(super) tx: Arc<Tx>,
    pub rate: u32,
}

impl AudioOutput {
    pub fn new() -> anyhow::Result<Self> {
        if let Ok(path) = std::env::var("NFS_AUDIO_DUMP") {
            DUMP.get_or_init(|| Mutex::new(std::fs::File::create(&path).expect("NFS_AUDIO_DUMP")));
            let (tx, _) = Tx::new(44100, true);
            tracing::warn!(path, "audio: offline dump (NFS_AUDIO_DUMP), no device");
            return Ok(Self { _stream: None, tx, rate: 44100 });
        }
        // TEST GLUE: scripted / test runs are silent — `NFS_MUTE=1`, or any `NFS_NO_FOCUS` run (the scripted
        // runs' marker) unless `NFS_SOUND=1` asks for the device (the audio tasks). No device is opened: the
        // mixer runs offline on the calling thread exactly as under `NFS_AUDIO_DUMP`, only nothing is written,
        // so the game's calls into the mixer are the same as with a device.
        let var = |k: &str| std::env::var_os(k).is_some_and(|v| !v.is_empty() && v != "0");
        if !var("NFS_SOUND") && (var("NFS_MUTE") || var("NFS_NO_FOCUS")) {
            let (tx, _) = Tx::new(44100, true);
            tracing::info!("audio: muted test run (NFS_MUTE / NFS_NO_FOCUS; NFS_SOUND=1 for the device), no device");
            return Ok(Self { _stream: None, tx, rate: 44100 });
        }
        let order = Backend::order_from_env();
        let mut errors = Vec::new();
        for (k, backend) in order.into_iter().enumerate() {
            let opened = match backend {
                Backend::Sdl => SdlStream::open().map(|(s, tx, rate)| Self { _stream: Some(Stream::Sdl(s)), tx, rate }),
                Backend::Cpal => Self::new_cpal(),
            };
            match opened {
                Ok(out) => {
                    if k > 0 {
                        tracing::warn!(?backend, wanted = ?order[0], "audio: fell back to the other backend");
                    }
                    return Ok(out);
                }
                Err(e) => {
                    tracing::warn!(?backend, error = %e, "audio: backend failed to open");
                    errors.push(format!("{backend:?}: {e}"));
                }
            }
        }
        Err(anyhow::anyhow!("no audio output ({})", errors.join("; ")))
    }

    /// The cpal device stream: the default device at its default config.
    fn new_cpal() -> anyhow::Result<Self> {
        let host = cpal::default_host();
        let device = host.default_output_device().ok_or_else(|| anyhow::anyhow!("no audio output device"))?;
        let supported = device.default_output_config()?;
        let rate = supported.sample_rate();
        let channels = supported.channels() as usize;
        // a smaller device buffer lowers the output latency and changes nothing in the mix: the
        // driver passes ([`Pump`]) and the bridge's queue run on the stream's own clock, whatever
        // size the callback asks for (`the_mix_does_not_depend_on_the_callback_size`)
        let mut config: cpal::StreamConfig = supported.config();
        let mut attempts = vec![cpal::BufferSize::Default];
        if let Some(n) = device_buffer_frames(&supported) {
            attempts.insert(0, cpal::BufferSize::Fixed(n));
        }
        let mut last = None;
        for size in attempts {
            config.buffer_size = size;
            let (tx, core) = Tx::new(rate, false);
            let core = core.expect("a device mixer has its callback half");
            match Self::open(&device, &config, channels, core) {
                Ok(stream) => {
                    stream.play()?;
                    tracing::info!(backend = "cpal", host = ?host.id(), rate, channels, buffer = ?config.buffer_size, "audio output started");
                    return Ok(Self { _stream: Some(Stream::Cpal(stream)), tx, rate });
                }
                Err(e) => {
                    tracing::warn!(error = %e, buffer = ?config.buffer_size, "audio stream refused");
                    last = Some(e);
                }
            }
        }
        Err(last.expect("an attempt was made").into())
    }

    /// TEST GLUE (`NFS_AUDIO_DUMP_TICKS=1`): one game tick of the offline mix, `rate / 64` frames at
    /// the mix rate appended to the dump — called once per `GameTick`; nothing otherwise.
    pub fn offline_game_tick(&self) {
        if let Some(o) = &self.tx.offline
            && dump_ticked()
        {
            o.lock().unwrap().game_tick();
        }
    }

    /// TEST GLUE: the offline mixer of a muted run (no device, nothing written), for the tests of
    /// the modules that drive it.
    #[cfg(test)]
    pub(crate) fn offline() -> Self {
        let (tx, _) = Tx::new(44100, true);
        Self { _stream: None, tx, rate: 44100 }
    }

    /// The output stream whose callback is `core`'s: what the game thread queued since the last
    /// run, then the mix — nothing here takes a lock or allocates.
    pub(super) fn open(device: &cpal::Device, config: &cpal::StreamConfig, channels: usize, mut core: Core) -> Result<cpal::Stream, cpal::Error> {
        let out_rate = config.sample_rate as f64;
        device.build_output_stream(
            config.clone(),
            move |out: &mut [f32], info: &cpal::OutputCallbackInfo| {
                core.drain();
                let ts = info.timestamp();
                // saturating to zero when the playback instant is not after the callback's (cpal 0.18)
                let delay = ts.playback.duration_since(ts.callback);
                let frames = (out.len() / channels.max(1)) as u64 + (delay.as_secs_f64() * out_rate).round() as u64;
                core.lat.frames.store(frames, Ordering::Relaxed);
                core.lat.at.store(core.lat.epoch.elapsed().as_nanos() as u64 + 1, Ordering::Release);
                core.fill(out, channels, out_rate);
                core.publish();
            },
            |e| tracing::error!(error = %e, "audio stream"),
            None,
        )
    }

    /// What 0x4A66B0 configures for `AUDIOQUALITY` when the sound opens (both start-ups 0x4A6840 /
    /// 0x4A6A00): the software voice count ([`voice_budget`]), the mix rate ([`mix_rate`]), the
    /// interpolator and the driver's sleep. The callers give it on every use; the same setting
    /// again does nothing (0x4A6A1C…0x4A6A2E: `[0x655D38]` unchanged → return).
    ///
    /// **A changed setting reopens the sound**, as 0x4A6A00 does (its one caller, the
    /// `Audio.Quality` / `Audio.Mode` commit 0x4E4820, 0x4E48B4): every voice stopped
    /// (0x5659F0(−1)), the music's stream destroyed (0x4AE2B0), the library closed (0x565890) and
    /// opened again at the new setting (0x4A66B0), its service clock from 0 (0x58EEA5…0x58EEB1).
    /// Returns `true` then: the caller makes what the reopen's tail makes — the music's stream and
    /// timer (0x4AE010), `fesfx` reloaded (0x4A8B80), `MusicPlay(NULL, 0)` (0x4AE4D0(0, 0),
    /// 0x4A6B1E). The first call configures the opened output (0x4A6840's open) and answers `false`.
    /// *(Until REG-AUDIO2 a change was applied under the running mix, the voices left playing.)*
    pub fn set_audio_quality(&self, audio_quality: u8) -> bool {
        let budget = voice_budget(audio_quality);
        let rate = mix_rate(audio_quality);
        let interp = Interp::for_quality(audio_quality);
        let sleep = pump_sleep_ms(audio_quality);
        let mut st = self.tx.state.lock().unwrap();
        if st.cfg_quality == Some(audio_quality) {
            return false;
        }
        let reopen = st.cfg_quality.is_some();
        st.cfg_quality = Some(audio_quality);
        // a new mix rate (or a reopen) drops the reverb (its delay lines are sized in frames of
        // the mix rate) and the next send makes a fresh one — made here, off the callback, and
        // handed over
        let reverb = (reopen || st.cfg_rate != Some(rate)).then(|| Box::new(ReverbTail::new(rate)));
        st.cfg_rate = Some(rate);
        self.tx.push(&mut st, Cmd::Quality { budget, rate, interp, sleep, reverb, reopen }, None);
        reopen
    }

    /// The aux byte of `Voice_play` for a collision one-shot: `voice+0x45 = 0x7F − aux`
    /// (0x4A9A5F…0x4A9A6C → 0x5906D7), so a loud hit sends less to the reverb.
    pub fn set_fx_aux(&self, id: VoiceId, aux: u8) {
        self.tx.send(Cmd::FxAux { id, aux });
    }

    /// `Voice_setVolume(h, 0)` **0x566340** over the voice table `0x656180…0x6563C8` (stride 8)
    /// — the race pause 0x4AB150 (0x4AB16A…0x4AB18A) and `I_SFX` (0x4AB4FB…0x4AB527) — and over
    /// the speech phrase's voices `[0x6573A8]` / `[0x6573AC]` (0x4AF900(0, 0), 0x4AF91D…0x4AF944).
    /// The voices keep running, silent: 0x5662B0 writes `+0x2C = 0` and zeroes the ramp step
    /// `+0x24` (0x566303…0x566306). The streams (music) are not in the table. **A released voice
    /// is not either**: every release forgets its handle (`0x656180[slot] = −1`, [`RELEASE_TICKS`]),
    /// so a voice on its way out ([`Ramp`] toward −1) runs its ramp to the stop, heard. The voices
    /// the audio update drives come back with its next `setVolume`; a one-shot, the speech phrase
    /// and a held horn stay silent to their end.
    ///
    /// **Settled 2026-09-27** (read to the instruction: 0x4AB150, 0x4A9C10 whole, 0x4A98E0 whole,
    /// 0x565E70/0x565EA0, 0x4AADB0, 0x4AAE30/0x4AAE50/0x4AAE70/0x4AAE90): the table's entry is 8
    /// bytes, `+0` the handle (or −1), `+4` the record/patch id last started in that slot, indexed
    /// by the **voice number** (`Voice_play`'s first argument, 0..0x7F) — not the bank slot. Only
    /// `Voice_play` 0x4A9C10 ever writes it, at 0x4a9de9 on a genuine (re)start, and only for a
    /// voice number other than 0x33: at 0x4a9cd4 `cmp ebx, 0x33 ; jne` sends 0x33 (the collision
    /// one-shot) to `Bank_playOneShot` too but reads the table back for its answer (0x4a9d0b)
    /// without writing it, since 0x33 is the one voice number every concurrent collision hit
    /// shares — writing it would make each hit steal the last one's handle. `Bank_playOneShot`
    /// 0x4A98E0 itself never touches the table (confirmed end to end); called straight from a
    /// caller with no voice number — the checkpoint chime 0x4AAE30, the wrong-way cue 0x4AAE50,
    /// the pylon-hit cue 0x4AAE70, the cop siren's start 0x4AAE90 — it is exactly as untracked as
    /// `Voice_start` 0x565E70's HUD (`fe_sound` 0x4AADB0) and front-end (`fesfx`) one-shots, which
    /// go through 0x565E70/0x565EA0's own voice pool and never call 0x4A9C10 or 0x4A98E0 at all.
    /// The port's `Voice` now carries this as `table` (see its field doc), set at every start and
    /// read by [`Shared::silence`], so the pause silences exactly the table's voices: the loops
    /// (engine layers, AI engines, road/wind/turbo, the squeal channels), the ambience and the
    /// player's horn, plus (folded in for the same effect as the speech's own 0x4AF900 stop) the
    /// announcer phrase — and leaves the HUD/front-end one-shots, the collision one-shots (wall,
    /// car-car, prop, bush, landing, crash) and the checkpoint chime / wrong-way cue / pylon-hit
    /// cue / cop siren sounding, as the original does. *(Until 2026-09-27 every voice, table or
    /// not, was silenced — a collision one-shot, a checkpoint chime or a front-end button click
    /// still ringing at the instant of the pause fell silent when it should not have.)*
    pub fn silence_voices(&self) {
        self.tx.send(Cmd::Silence);
    }

    /// A playing voice's pan from the caller's 16-bit angle: the pan byte ([`pan_byte`]) to
    /// `Voice_setPan8` 0x566100, which stores it in `voice+0x1F` (and does nothing when it is the
    /// same, 0x56616D).
    pub fn set_pan(&self, id: VoiceId, pan: u16) {
        self.tx.send(Cmd::Pan { id, pan });
    }

    /// Fraction of the clip already played (0..1), or None when the voice is gone; 0 for a held
    /// anchor ([`status_voice`](Self::status_voice)).
    pub fn progress(&self, id: VoiceId) -> Option<f32> {
        let (pend, snap) = self.tx.overlay(id);
        match pend {
            // started, not yet in the mix: at its beginning
            Some(Pend::Start(_)) => Some(0.0),
            Some(_) => None,
            None => snap.status_voice(id).map(|v| {
                if v.held {
                    return 0.0;
                }
                let n = v.frames.max(1) as f64;
                (v.position / n).clamp(0.0, 1.0) as f32
            }),
        }
    }

    /// The voice's position in ms — what the sound library's status call 0x566910 answers at `+4`:
    /// the frames played `· 1000 / rate` (0x5669B9…0x5669D5, a 64-bit multiply and divide), 0 for
    /// a held anchor ([`status_voice`](Self::status_voice)), or `None` once the voice has ended
    /// (its state 3, 0x566A1E).
    pub fn position_ms(&self, id: VoiceId) -> Option<i32> {
        let (pend, snap) = self.tx.overlay(id);
        match pend {
            Some(Pend::Start(_)) => Some(0),
            Some(_) => None,
            None => snap.status_voice(id).map(|v| if v.held { 0 } else { played_ms(v.position as u64, v.rate) }),
        }
    }

    /// **`Voice_play` 0x4A9C10's start** (`Voice_start` 0x565E70 → 0x5907B0 → 0x590420): the
    /// library units, no float in between — the pitch **word** `+0xC` ([`resample::PITCH_UNIT`] =
    /// 4096 the sample's root, `pitch16_16 >> 4`, 0x4A9AC5), the **bend byte** `+9` (`voice+0x3F`)
    /// and the pan byte `+7` ([`pan_byte`] of the angle `pan`), taken together. `sel` (0x7F here,
    /// the request's default) is `Voice_play`'s seventh argument (0x4A9B89…0x4A9B90): the start
    /// (0x5907B0) tests each record's codes 0x01 / 0x02 against it and multiplies the patch volume
    /// by `sel / 127` (0x5905D0), and draws from the library's generator
    /// ([`crate::librand::patch_start`]) in the original's order. `clip.channels` voices of the
    /// budget go through the allocator ([`pick_victims`]); a refused start still returns a handle,
    /// one that names no voice — the library's −9, on which every later call fails
    /// ([`is_playing`](Self::is_playing) is false).
    ///
    /// Every record whose ranges hold `sel` starts as a voice of its own, with its own sample
    /// ([`Clip::layers`]), its own allocation (0x590420 → 0x5898F0 with the record's channel count
    /// and priority) and its own draws, in record order (0x590841…0x590992). A refused allocation
    /// stops the records already started and the start fails (0x5909A6). Two or more started make
    /// a **group** (0x5909DB…0x590A5C): one id (`voice+0x1E` from 0x5903D0), the last started the
    /// anchor (`+0x1B`), and the start answers the anchor's handle (`[esp+0x14]`, 0x590420's last
    /// answer). Every later call on the handle walks the group (0x590340: stop 0x565D60, volume,
    /// pitch, pan), which here is every voice with the returned [`VoiceId`]. *(Until 2026-09-27 the
    /// port played one merged sample per patch, the last record's, with the last record's draws —
    /// an approximation; until 2026-09-27, later, audio port 8, every caller gave a float pitch
    /// factor here instead of the word, [`resample::PITCH_UNIT`]'s own doc.)*
    pub fn play_word(&self, clip: Arc<Clip>, volume: f32, word: u16, bend: u8, pan: u16, looping: bool) -> VoiceId {
        self.play_word_sel(clip, volume, word, bend, pan, looping, 0x7F)
    }

    /// [`play_word`](Self::play_word) with the request's byte `+6`, `sel` — the collision one-shot's
    /// own aux/gate byte, the only shipped caller of a `sel` other than 0x7F.
    pub fn play_word_sel(&self, clip: Arc<Clip>, volume: f32, word: u16, bend: u8, pan: u16, looping: bool, sel: u8) -> VoiceId {
        self.play_word_tabled(clip, volume, word, bend, pan, looping, sel, true)
    }

    /// [`play_word`](Self::play_word) for a start the original makes with a bare `Voice_start`
    /// 0x565E70, never `Voice_play` 0x4A9C10: the HUD's `fe_sound` (0x4AADB0), the front end's
    /// `fesfx`, the checkpoint chime, the wrong-way cue and the pylon-hit cue (straight to
    /// `Bank_playOneShot` 0x4A98E0, no voice number at all). Not walked by
    /// [`silence_voices`](Self::silence_voices).
    pub fn play_word_untabled(&self, clip: Arc<Clip>, volume: f32, word: u16, bend: u8, pan: u16, looping: bool) -> VoiceId {
        self.play_word_tabled(clip, volume, word, bend, pan, looping, 0x7F, false)
    }

    /// [`play_word_untabled`](Self::play_word_untabled) with the request's byte `+6`
    /// ([`play_word_sel`](Self::play_word_sel)): the collision one-shot's own path, voice 0x33 —
    /// `Voice_play` takes it to `Bank_playOneShot` (0x4a9cd4…0x4a9d0b) but never writes the table
    /// for it (see the `table` field of [`Voice`]).
    pub fn play_word_sel_untabled(&self, clip: Arc<Clip>, volume: f32, word: u16, bend: u8, pan: u16, looping: bool, sel: u8) -> VoiceId {
        self.play_word_tabled(clip, volume, word, bend, pan, looping, sel, false)
    }

    pub(super) fn play_word_tabled(&self, clip: Arc<Clip>, volume: f32, word: u16, bend: u8, pan: u16, looping: bool, sel: u8, tabled: bool) -> VoiceId {
        self.play(PlayArgs { clip, volume, word, bend, pan, looping, sel, tabled, filter: None })
    }

    /// Every `play_word*` start in one call: the [`PlayArgs`] are the request the wrappers fill.
    pub fn play(&self, args: PlayArgs) -> VoiceId {
        let PlayArgs { clip, volume, word, bend, pan, looping, sel, tabled, filter } = args;
        let mut st = self.tx.state.lock().unwrap();
        let id = VoiceId(st.next_id);
        st.next_id += 1;
        self.tx.push(&mut st, Cmd::Start { id, clip, volume, word, bend, pan, looping, sel, tabled, filter }, Some(Pend::Start(id)));
        id
    }

    /// `Voice_play`'s update of a playing voice (0x4A9E1F…0x4A9FA3): `setVolume`, `Voice_setBend`
    /// 0x5660C0 with the byte, `Voice_setPitch` 0x565F80 with the word (`pitch16_16 >> 4`, the low
    /// 16 bits into `voice+0x6C`, 0x565FEB) — the pan is [`set_pan`](Self::set_pan).
    pub fn set_word(&self, id: VoiceId, volume: f32, word: u16, bend: u8) {
        self.tx.send(Cmd::SetWord { id, volume, word, bend });
    }

    /// **0x565DD0(handle, ticks, target)** on every playing voice of the handle's group
    /// ([`Ramp`]); a held anchor (state 2) is not walked (0x590340 takes state 1 only).
    pub fn ramp(&self, id: VoiceId, ticks: i32, target: i32) {
        self.tx.send(Cmd::Ramp { id, ticks, target });
    }

    /// **0x565F00(handle, value)**, the strip hook `+0xE0` on every playing voice of the group
    /// ([`LowPass`]); `value` from [`voice_play_filter`].
    pub fn set_filter(&self, id: VoiceId, value: i32) {
        self.tx.send(Cmd::Filter { id, value });
    }

    /// `Voice_play`'s update tail (0x4A9FA8…0x4AA03B): [`set_filter`](Self::set_filter) with
    /// `value` when there is one.
    pub fn voice_play_filter(&self, id: VoiceId, value: Option<i32>) {
        if let Some(v) = value {
            self.set_filter(id, v);
        }
    }

    /// The audio update's release of a voice, 0x565DD0(h, [`RELEASE_TICKS`], −1) — 0x4A9890 and
    /// the direct calls ([`RELEASE_TICKS`]'s list). The caller drops the handle.
    pub fn release(&self, id: VoiceId) {
        self.ramp(id, RELEASE_TICKS, RELEASE_TARGET);
    }

    /// `Voice_play`'s release of a slot's previous sample, 0x565DD0(h, 0x19, −1) (0x4A9DA4).
    pub fn release_replaced(&self, id: VoiceId) {
        self.ramp(id, REPLACE_TICKS, RELEASE_TARGET);
    }

    pub fn stop(&self, id: VoiceId) {
        self.tx.stop(id);
    }

    /// A cheap, cloneable handle that can [`Self::stop`] a voice without keeping the whole
    /// `AudioOutput` (its device stream) alive — `FEMovie`'s own audio track (`mad.md` §14) needs
    /// to stop itself well after the code that started it has moved on: at a loop restart
    /// (0x4E2380's `0x567690`) and whenever its widget is torn down (the destructor was not traced;
    /// this is the port's own equivalent, so no voice a widget started is ever left playing once
    /// the widget is gone).
    pub fn stop_handle(&self) -> StopHandle {
        StopHandle(self.tx.clone())
    }

    /// `0x565CD0(id, level)`: bit 0x20 of the id sets the bus `[0x6B4A0C]`, bit 0x80
    /// `[0x6B4A0D]` (an id with none of 0xE0 sets all, 0x565CDD…0x565CE1), stored as bytes; the
    /// voices' sends follow at once (0x565D28…0x565D51 → 0x5902A0). `p20` / `p80` are the tunnel
    /// reverb's two parameters (`reverb::EnvParams`).
    pub fn set_env(&self, p20: i32, p80: i32) {
        self.tx.bus[0].store(p20 as i8, Ordering::Relaxed);
        self.tx.bus[1].store(p80 as i8, Ordering::Relaxed);
        self.tx.send(Cmd::Env { p20, p80 });
    }

    /// The two buses as `set_env` left them.
    pub fn env(&self) -> (i8, i8) {
        (self.tx.bus[0].load(Ordering::Relaxed), self.tx.bus[1].load(Ordering::Relaxed))
    }

    /// **0x56AFA0** → 0x592E40 → 0x58F5D0, the driver's latency in ms: the software mixer's write
    /// position `[0x6AD78C]` less the primary buffer's play cursor (`GetCurrentPosition`, bytes
    /// `/ 2 / channels`), wrapped by the buffer length `[0x6AD7B4]`, `· 1000 / rate` (`[0x6B48DA]`,
    /// `idiv`) — what has been mixed and not yet heard; 0 with no buffer or on a failed call. Here:
    /// what the last callback queued ([`latency_ms`]) less the time since.
    pub fn latency_ms(&self) -> i32 {
        let l = &self.tx.lat;
        let at = l.at.load(Ordering::Acquire);
        if at == 0 {
            return 0;
        }
        let since = l.epoch.elapsed().saturating_sub(std::time::Duration::from_nanos(at - 1));
        latency_ms(l.frames.load(Ordering::Relaxed), since, self.rate)
    }

    /// Is a one-shot / loop voice still playing?
    pub fn is_playing(&self, id: VoiceId) -> bool {
        let (pend, snap) = self.tx.overlay(id);
        match pend {
            Some(Pend::Start(_) | Pend::Stream(..)) => true,
            Some(Pend::Stop(_)) => false,
            None => snap.voices.iter().any(|v| v.id == id && !v.finished) || snap.streams.iter().any(|(i, _)| *i == id),
        }
    }
}
