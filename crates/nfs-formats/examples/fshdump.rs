//! fshdump <file.fsh> <out_dir> — every entry as a PPM-free raw RGBA PAM (tooling)
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let f = nfs_formats::fsh::Fsh::parse(&data).unwrap();
    for i in 0..f.entries.len() {
        let Ok(img) = f.decode(&data, i) else { continue };
        let mut out = format!("P7\nWIDTH {}\nHEIGHT {}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n", img.width, img.height).into_bytes();
        out.extend_from_slice(&img.rgba);
        std::fs::write(format!("{}/{:02}.pam", a[2], i), out).unwrap();
    }
    println!("{} entries", f.entries.len());
}
