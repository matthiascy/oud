use anyhow::Result;
use std::env;

fn main() -> Result<()> {
    // Rerun the build script if the contents of assets/ changes.
    println!("cargo:rerun-if-changed=assets/*");

    // Copy the assets to the target directory
    let mut copy_options = fs_extra::dir::CopyOptions::new();
    copy_options.overwrite = true;

    let paths_to_copy = vec!["assets/"];
    let out_dir = env::var("OUT_DIR")?;
    fs_extra::copy_items(&paths_to_copy, out_dir, &copy_options)?;

    Ok(())
}
