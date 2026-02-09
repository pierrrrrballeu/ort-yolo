use ort::session::Session;
use ort::ep::CPUExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;

#[test]
fn model_info() -> ort::Result<()> {
    let model_path = r#"C:\Users\xonbi\Desktop\project\python\sora-wmr\models\yolo11n.onnx"#;
    
    let model = Session::builder()?
        .with_execution_providers([ CPUExecutionProvider::default().build() ])?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(model_path)?;
    
    for input in model.inputs() {
        println!("Input {} Type {}", input.name(), input.dtype());
    }
    
    for output in model.outputs() {
        println!("Output {} Type {}", output.name(), output.dtype())
    }
    
    Ok(())
}