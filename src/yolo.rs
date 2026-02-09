use anyhow::Error;
use image::{DynamicImage, GenericImageView, Rgba};
use imageproc::drawing::{draw_hollow_rect, draw_hollow_rect_mut};
use imageproc::rect::Rect;
use ndarray::{Array, Axis};
use ort::ep::CPUExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionOutputs};
use ort::value::TensorRef;

const YOLOV8_CLASS_LABELS: [&str; 80] = [
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
	"bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
	"carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
	"tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

pub fn predict(model_path: &str, img: DynamicImage) -> anyhow::Result<DynamicImage, Error> {
    let mut yolo = Session::builder()?
        .with_execution_providers(&[CPUExecutionProvider::default().build()])?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(model_path)?;

    let mut input = Array::zeros((1, 3, 640, 640));
    
    let mut resized_image = preprocess(&img, &mut input);
    let output = yolo.run(inputs!["images" => TensorRef::from_array_view(&input)?])?;

    postprocess(output, &mut resized_image)?;
    
    Ok(resized_image)
}

fn preprocess(
    image: &DynamicImage,
    input: &mut ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>, f32>,
) -> DynamicImage {
    let resized_image = image.resize_exact(640, 640, image::imageops::FilterType::CatmullRom);
    
    for p in resized_image.pixels() {
        let x = p.0 as _;
        let y = p.1 as _;
        let [r, g, b, _] = p.2.0;
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }
    
    resized_image
}

struct BBox {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

fn postprocess(output: SessionOutputs, img: &mut DynamicImage) -> ort::Result<()> {
    let out = output["output0"]
        .try_extract_array::<f32>()?
        .index_axis(Axis(0), 0)
        .reversed_axes()
        .into_owned();

    for i in out.axis_iter(Axis(0)) {
        let scores = i.slice(ndarray::s![4..]);
        let class_id = 0; // person

        if scores[class_id] >= 0.5 {
            let coords = i.slice(ndarray::s![..4]);

            let cx = coords[0];
            let cy = coords[1];
            let w = coords[2];
            let h = coords[3];

            let x = cx - w / 2.0;
            let y = cy - h / 2.0;
            
            let rect = Rect::at(x as i32, y as i32).of_size(w as u32, h as u32);
            draw_hollow_rect_mut(img, rect, Rgba::<u8>([255, 0, 0, 0]));
        }
    }

    Ok(())
}
