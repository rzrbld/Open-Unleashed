//! pic16check <ALLdrivr.fsh> <index> <original pic16.fsh> — the port's pic16 against the original's file
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let all = std::fs::read(&a[1]).unwrap();
    let ours = nfs_game::player_picture::pic16_fsh(&all, a[2].parse().unwrap()).unwrap();
    let orig = std::fs::read(&a[3]).unwrap();
    let diff = ours.iter().zip(&orig).filter(|(x, y)| x != y).count();
    let first = ours.iter().zip(&orig).position(|(x, y)| x != y);
    println!("ours {} bytes, original {} bytes, differing bytes {diff}, first at {first:?}", ours.len(), orig.len());
}
