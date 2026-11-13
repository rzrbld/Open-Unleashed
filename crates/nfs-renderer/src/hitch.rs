//! ReNFS5: the hitch log (`NFS5_HITCH_LOG=1`): phase marks through a frame, and for any frame over
//! `NFS5_HITCH_MS` (40 ms) one line with where its time went — for finding the stutter.

use std::cell::RefCell;
use std::time::Instant;

struct State {
    start: Option<Instant>,
    last: Option<Instant>,
    marks: Vec<(&'static str, f32)>,
    note: String,
    counts: Vec<(&'static str, u32)>,
}

thread_local! {
    static S: RefCell<State> = const { RefCell::new(State { start: None, last: None, marks: Vec::new(), note: String::new(), counts: Vec::new() }) };
}

fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NFS5_HITCH_LOG").is_ok_and(|v| v.trim() == "1"))
}

/// The time from the last mark to now, under `name`.
pub fn mark(name: &'static str) {
    if !on() {
        return;
    }
    S.with(|s| {
        let mut s = s.borrow_mut();
        let now = Instant::now();
        if let Some(l) = s.last {
            let ms = (now - l).as_secs_f32() * 1000.0;
            s.marks.push((name, ms));
        }
        s.last = Some(now);
    });
}

/// A note on this frame (e.g. how many sim ticks ran).
pub fn note(text: &str) {
    if !on() {
        return;
    }
    S.with(|s| {
        let mut s = s.borrow_mut();
        if !s.note.is_empty() {
            s.note.push(' ');
        }
        s.note.push_str(text);
    });
}

/// One more of `name` this frame (shown as `name=n` in the frame's line).
pub fn count(name: &'static str) {
    if !on() {
        return;
    }
    S.with(|s| {
        let mut s = s.borrow_mut();
        match s.counts.iter_mut().find(|c| c.0 == name) {
            Some(c) => c.1 += 1,
            None => s.counts.push((name, 1)),
        }
    });
}

/// The frame starts: the time since the last frame's end is `wait`; a slow previous frame is logged.
pub fn frame_start() {
    if !on() {
        return;
    }
    S.with(|s| {
        let mut s = s.borrow_mut();
        let now = Instant::now();
        if let Some(st) = s.start {
            let total = (now - st).as_secs_f32() * 1000.0;
            let limit: f32 = std::env::var("NFS5_HITCH_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(40.0);
            if let Some(l) = s.last {
                let ms = (now - l).as_secs_f32() * 1000.0;
                s.marks.push(("wait", ms));
            }
            if total > limit {
                // the same phase several times in a frame (one per command) is summed, with its count
                let mut sums: Vec<(&str, f32, u32)> = Vec::new();
                for (n, ms) in &s.marks {
                    match sums.iter_mut().find(|e| e.0 == *n) {
                        Some(e) => {
                            e.1 += ms;
                            e.2 += 1;
                        }
                        None => sums.push((n, *ms, 1)),
                    }
                }
                let parts: Vec<String> = sums.iter().filter(|e| e.1 >= 0.5).map(|(n, ms, c)| if *c > 1 { format!("{n}={ms:.1}x{c}") } else { format!("{n}={ms:.1}") }).collect();
                let thread = std::thread::current().name().unwrap_or("?").to_string();
                for (n, c) in std::mem::take(&mut s.counts) {
                    let t = format!(" {n}={c}");
                    s.note.push_str(&t);
                }
                tracing::info!(thread, total_ms = format!("{total:.1}"), phases = parts.join(" "), note = %s.note, "hitch");
            }
        }
        s.marks.clear();
        s.note.clear();
        s.counts.clear();
        s.start = Some(now);
        s.last = Some(now);
    });
}
