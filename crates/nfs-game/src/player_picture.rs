//! The player's picture as the network race sends it: `SaveData\pic16.fsh`.
//!
//! The original keeps that file for the profile on show (vtable 0x5B6A98, its setter 0x4DBBA0): when
//! the picture changes it takes entry `n` of `FEData\Art\AllDrivr.fsh` (0x4DBCE1…0x4DBD23, `n` the
//! leading digit of the profile's file name, the FE's own pick) — or a custom image beside it,
//! converted by `gimme.dll`'s `ConvertFile`, which this port does not take — and writes it as a
//! one-entry FSH: the entry's own name, 54 × 64, **16-bit 565** (code 0x78). The race load 0x427A60
//! then reads `%s%s.fsh` over the save directory `[0x65B360]` and `pic16` (0x5CCCC8, 0x5CCCE0) and
//! posts it as file type 0 — the picture the other players' `HLeader` and loading rows show.
//! *(Until 2026-11-13 the port posted the shipped `GameData\DashHud\pic16.fsh`, the default face,
//! for every player.)*

use nfs_formats::fsh::Fsh;

/// `SaveData\pic16.fsh` for picture `index` of `AllDrivr.fsh` (its bytes), the file the original writes.
pub fn pic16_fsh(alldrivr: &[u8], index: usize) -> Option<Vec<u8>> {
    let f = Fsh::parse(alldrivr).ok()?;
    let entry = f.entries.get(index)?;
    let img = f.decode(alldrivr, index).ok()?;
    let (w, h) = (img.width as usize, img.height as usize);
    if w == 0 || h == 0 || img.rgba.len() < w * h * 4 {
        return None;
    }
    let mut name = [b' '; 4];
    for (d, s) in name.iter_mut().zip(entry.name.bytes()) {
        *d = s;
    }
    let total = 32 + 16 + w * h * 2;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"SHPI");
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(b"GIMX");
    out.extend_from_slice(&name);
    out.extend_from_slice(&32u32.to_le_bytes());
    out.extend_from_slice(b"Buy ERTS");
    // the image block: code 0x78 (16-bit 565), no next block, width, height, the four misc words 0
    out.extend_from_slice(&[0x78, 0, 0, 0]);
    out.extend_from_slice(&(w as u16).to_le_bytes());
    out.extend_from_slice(&(h as u16).to_le_bytes());
    out.extend_from_slice(&[0u8; 8]);
    for px in img.rgba.chunks_exact(4).take(w * h) {
        let v = (u16::from(px[0] >> 3) << 11) | (u16::from(px[1] >> 2) << 5) | u16::from(px[2] >> 3);
        out.extend_from_slice(&v.to_le_bytes());
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    /// The original's own `SaveData\pic16.fsh` for profile `1PAULO` (picture 1, `M02`), byte for byte.
    #[test]
    fn picture_one_is_the_originals_pic16() {
        let Some(root) = nfs_assets::platform::dev_game_root() else { return };
        let (Some(all), Some(orig)) = (std::fs::read(root.join("FEData/Art/ALLdrivr.fsh")).ok(), std::fs::read(root.join("SaveData/pic16.fsh")).ok()) else { return };
        assert_eq!(super::pic16_fsh(&all, 1).as_deref(), Some(orig.as_slice()));
    }
}
