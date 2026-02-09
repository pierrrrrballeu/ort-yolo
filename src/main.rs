#[cfg(test)]
mod tests;
mod yolo;

use anyhow::Error;

fn main() -> anyhow::Result<(), Error> {
    let img = image::open("images/bus.jpg")?;
    
    let result = yolo::predict(
        r#"models\yolo11n.onnx"#,
        img,
    )?;
    
    result.save("result.png")?;

    Ok(())
}
