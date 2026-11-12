//! SDL2 scancodes (USB HID usages) and keycodes → winit's physical and logical keys.

use crate::keyboard::{Key, KeyCode, KeyLocation, NamedKey, NativeKeyCode, PhysicalKey, SmolStr};

pub(crate) fn physical(sc: u32) -> PhysicalKey {
    use KeyCode::*;
    const LETTERS: [KeyCode; 26] = [
        KeyA, KeyB, KeyC, KeyD, KeyE, KeyF, KeyG, KeyH, KeyI, KeyJ, KeyK, KeyL, KeyM, KeyN, KeyO, KeyP, KeyQ, KeyR, KeyS,
        KeyT, KeyU, KeyV, KeyW, KeyX, KeyY, KeyZ,
    ];
    const DIGITS: [KeyCode; 10] = [Digit1, Digit2, Digit3, Digit4, Digit5, Digit6, Digit7, Digit8, Digit9, Digit0];
    const FKEYS: [KeyCode; 12] = [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12];
    const PAD: [KeyCode; 10] = [Numpad1, Numpad2, Numpad3, Numpad4, Numpad5, Numpad6, Numpad7, Numpad8, Numpad9, Numpad0];
    const FHI: [KeyCode; 12] = [F13, F14, F15, F16, F17, F18, F19, F20, F21, F22, F23, F24];
    let code = match sc {
        4..=29 => LETTERS[(sc - 4) as usize],
        30..=39 => DIGITS[(sc - 30) as usize],
        40 => Enter,
        41 => Escape,
        42 => Backspace,
        43 => Tab,
        44 => Space,
        45 => Minus,
        46 => Equal,
        47 => BracketLeft,
        48 => BracketRight,
        49 | 50 => Backslash,
        51 => Semicolon,
        52 => Quote,
        53 => Backquote,
        54 => Comma,
        55 => Period,
        56 => Slash,
        57 => CapsLock,
        58..=69 => FKEYS[(sc - 58) as usize],
        70 => PrintScreen,
        71 => ScrollLock,
        72 => Pause,
        73 => Insert,
        74 => Home,
        75 => PageUp,
        76 => Delete,
        77 => End,
        78 => PageDown,
        79 => ArrowRight,
        80 => ArrowLeft,
        81 => ArrowDown,
        82 => ArrowUp,
        83 => NumLock,
        84 => NumpadDivide,
        85 => NumpadMultiply,
        86 => NumpadSubtract,
        87 => NumpadAdd,
        88 => NumpadEnter,
        89..=98 => PAD[(sc - 89) as usize],
        99 => NumpadDecimal,
        100 => IntlBackslash,
        101 => ContextMenu,
        102 => Power,
        103 => NumpadEqual,
        104..=115 => FHI[(sc - 104) as usize],
        224 => ControlLeft,
        225 => ShiftLeft,
        226 => AltLeft,
        227 => SuperLeft,
        228 => ControlRight,
        229 => ShiftRight,
        230 => AltRight,
        231 => SuperRight,
        _ => return PhysicalKey::Unidentified(NativeKeyCode::Xkb(sc)),
    };
    PhysicalKey::Code(code)
}

pub(crate) fn location(sc: u32) -> KeyLocation {
    match sc {
        84..=99 | 103 => KeyLocation::Numpad,
        224..=227 => KeyLocation::Left,
        228..=231 => KeyLocation::Right,
        _ => KeyLocation::Standard,
    }
}

/// The logical key: a named key for the specials (by position), else the character the layout
/// gives (`sym`, SDL's keycode — the Unicode value for a printable key), upper case with Shift.
pub(crate) fn logical(sc: u32, sym: i32, shift: bool) -> (Key, Option<SmolStr>) {
    use NamedKey::*;
    let named = match sc {
        40 | 88 => Some(Enter),
        41 => Some(Escape),
        42 => Some(Backspace),
        43 => Some(Tab),
        44 => Some(Space),
        57 => Some(CapsLock),
        58 => Some(F1),
        59 => Some(F2),
        60 => Some(F3),
        61 => Some(F4),
        62 => Some(F5),
        63 => Some(F6),
        64 => Some(F7),
        65 => Some(F8),
        66 => Some(F9),
        67 => Some(F10),
        68 => Some(F11),
        69 => Some(F12),
        70 => Some(PrintScreen),
        71 => Some(ScrollLock),
        72 => Some(Pause),
        73 => Some(Insert),
        74 => Some(Home),
        75 => Some(PageUp),
        76 => Some(Delete),
        77 => Some(End),
        78 => Some(PageDown),
        79 => Some(ArrowRight),
        80 => Some(ArrowLeft),
        81 => Some(ArrowDown),
        82 => Some(ArrowUp),
        83 => Some(NumLock),
        101 => Some(ContextMenu),
        224 | 228 => Some(Control),
        225 | 229 => Some(Shift),
        226 | 230 => Some(Alt),
        227 | 231 => Some(Super),
        _ => None,
    };
    if let Some(n) = named {
        let text = match n {
            Enter => Some(SmolStr::new("\r")),
            Space => Some(SmolStr::new(" ")),
            Tab => Some(SmolStr::new("\t")),
            _ => None,
        };
        return (Key::Named(n), text);
    }
    if (0x20..0x4000_0000).contains(&sym) && sym != 0x7F {
        if let Some(c) = char::from_u32(sym as u32) {
            let c = if shift { c.to_uppercase().next().unwrap_or(c) } else { c };
            let s = SmolStr::new(c.encode_utf8(&mut [0; 4]));
            return (Key::Character(s.clone()), Some(s));
        }
    }
    (Key::Unidentified(crate::keyboard::NativeKey::Xkb(sc)), None)
}
