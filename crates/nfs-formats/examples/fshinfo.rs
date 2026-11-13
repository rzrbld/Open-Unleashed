//! fshinfo <file.fsh> — entries: name, size, format, block length (tooling)
fn main() {
    let path = std::env::args().nth(1).expect("path");
    let data = std::fs::read(&path).unwrap();
    let f = nfs_formats::fsh::Fsh::parse(&data).unwrap();
    println!("{} bytes, {} entries", data.len(), f.entries.len());
    for i in 0..f.entries.len() {
        let name = f.entry_caption(&data, i).unwrap_or_default();
        let e = &f.entries[i];
        let img = f.decode(&data, i).ok();
        println!("{i}: {:?} {:?} {}x{} block={}", e, name, img.as_ref().map_or(0, |m| m.width), img.as_ref().map_or(0, |m| m.height), f.image_block_len(data.len(), i));
    }
}
