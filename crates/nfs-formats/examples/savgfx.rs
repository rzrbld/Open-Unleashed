//! savgfx <file.sav> — the profile's Graphics.* / Display.* accessors (tooling)
fn main() {
    let path = std::env::args().nth(1).expect("sav path");
    let data = std::fs::read(path).unwrap();
    let s = nfs_formats::sav::SaveFile::parse(&data).unwrap();
    let Some(a) = s.accessors else { println!("no accessor block"); return };
    for (name, v) in a.entries.iter() {
        if name.starts_with("Graphics.") || name.starts_with("Display.") {
            println!("{name} = {v:?}");
        }
    }
}
