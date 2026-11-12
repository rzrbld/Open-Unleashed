//! TEST GLUE: opens the device output (`NFS_AUDIO` = sdl | cpal | auto picks the backend) and
//! plays silence for half a second, logging what was opened.
fn main() {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let out = nfs_audio::AudioOutput::new().expect("audio output");
    std::thread::sleep(std::time::Duration::from_millis(500));
    println!("opened at {} Hz, latency {} ms", out.rate, out.latency_ms());
}
