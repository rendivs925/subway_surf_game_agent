use anyhow::Result;
use once_cell::sync::Lazy;
use onnxruntime::{environment::Environment, LoggingLevel};
use opencv::prelude::*;
use opencv::{core, imgproc, videoio};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tokio::process::Command;
use tokio::time::sleep;
// Import configuration (assumed to exist)
use subway_surfers_bot::config::{apply_profile, with_config};

static ORT_ENV: Lazy<Environment> = Lazy::new(|| {
    Environment::builder()
        .with_name("subway_surfers_detector")
        .with_log_level(LoggingLevel::Warning)
        .build()
        .expect("Failed to create ONNX Runtime environment")
});

// COCO class names for YOLO
fn coco80() -> Vec<String> {
    vec![
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect()
}

// Map COCO labels to game-specific classes
fn map_coco_to_game(label: &str) -> Option<&'static str> {
    match label {
        "person" => Some("player"),
        "train" | "bus" | "truck" => Some("train_blocking"),
        "bench" | "couch" | "chair" | "dining table" => Some("barrier_ground"),
        "stop sign" | "tv" => Some("barrier_overhead"),
        _ => None,
    }
}

struct Letterbox {
    scale: f32,
    pad_x: f32,
    pad_y: f32,
}

fn letterbox_bgr_to_rgb_nchw(
    frame: &core::Mat,
    new_size: i32,
) -> Result<(ndarray::Array4<f32>, Letterbox)> {
    let (w, h) = (frame.cols() as f32, frame.rows() as f32);
    let s = (new_size as f32 / w).min(new_size as f32 / h);
    let nw = (w * s).round() as i32;
    let nh = (h * s).round() as i32;
    let px = ((new_size - nw) / 2).max(0);
    let py = ((new_size - nh) / 2).max(0);

    let mut resized = core::Mat::default();
    imgproc::resize(
        frame,
        &mut resized,
        core::Size::new(nw, nh),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    let padded = core::Mat::zeros(new_size, new_size, frame.typ())?;
    let roi = core::Rect::new(px, py, nw, nh);
    let padded_mat = padded.to_mat()?;
    let roi_ref = core::Mat::roi(&padded_mat, roi)?;
    let mut dst_roi = roi_ref.try_clone()?;
    resized.copy_to(&mut dst_roi)?;

    let mut rgb = core::Mat::default();
    imgproc::cvt_color(
        &padded,
        &mut rgb,
        imgproc::COLOR_BGR2RGB,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let size = rgb.size()?;
    let mut arr = ndarray::Array4::<f32>::zeros((1, 3, size.height as usize, size.width as usize));

    unsafe {
        let data = rgb.ptr(0)? as *const u8;
        for y in 0..size.height as usize {
            for x in 0..size.width as usize {
                let i = (y * size.width as usize + x) * 3;
                let r = *data.add(i) as f32 / 255.0;
                let g = *data.add(i + 1) as f32 / 255.0;
                let b = *data.add(i + 2) as f32 / 255.0;
                arr[[0, 0, y, x]] = r;
                arr[[0, 1, y, x]] = g;
                arr[[0, 2, y, x]] = b;
            }
        }
    }

    Ok((
        arr,
        Letterbox {
            scale: s,
            pad_x: px as f32,
            pad_y: py as f32,
        },
    ))
}

mod vision {
    use super::*;
    use onnxruntime::session::Session;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Detection {
        pub class_name: String,
        pub confidence: f32,
        pub bbox: BoundingBox,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BoundingBox {
        pub x: f32,
        pub y: f32,
        pub width: f32,
        pub height: f32,
    }

    pub struct FrameCapture {
        cap: videoio::VideoCapture,
        source: String,
        width: f32,
        height: f32,
        max_retries: u32,
        retry_delay: Duration,
    }

    impl FrameCapture {
        pub fn new() -> Result<Self> {
            let max_retries = 3;
            let retry_delay = Duration::from_millis(100);

            let devices = [(2, "/dev/video2"), (3, "/dev/video3"), (0, "/dev/video0")];
            let mut cap: Option<videoio::VideoCapture> = None;
            let mut source = String::new();

            for (device_id, device_name) in devices.iter() {
                println!("🔍 Trying V4L2 device: {}", device_name);
                match videoio::VideoCapture::new(*device_id, videoio::CAP_V4L2) {
                    Ok(mut c) if c.is_opened()? => {
                        // Set optimal properties but don't force specific resolution
                        c.set(videoio::CAP_PROP_FPS, 90.0)?;
                        c.set(videoio::CAP_PROP_BUFFERSIZE, 1.0)?;

                        let width = c.get(videoio::CAP_PROP_FRAME_WIDTH)? as f32;
                        let height = c.get(videoio::CAP_PROP_FRAME_HEIGHT)? as f32;

                        // Accept any reasonable resolution (minimum 480p)
                        if width >= 480.0 && height >= 360.0 {
                            cap = Some(c);
                            source = format!("{} (scrcpy v4l2loopback)", device_name);
                            println!("✅ Using device {} with resolution {}x{}", device_name, width, height);
                            break;
                        } else {
                            println!(
                                "⚠️  Device {} resolution {}x{} too small (minimum 480x360)",
                                device_name, width, height
                            );
                        }
                    }
                    Ok(_) => {
                        println!("⚠️  Device {} opened but not usable", device_name);
                    }
                    Err(e) => {
                        println!("⚠️  Failed to open {}: {}", device_name, e);
                    }
                }
            }

            let cap = cap.ok_or_else(|| {
                anyhow::anyhow!(
                    "Failed to open any V4L2 device. Ensure scrcpy is running with --v4l2-sink=/dev/video2 --fullscreen --max-fps=90."
                )
            })?;

            let width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as f32;
            let height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as f32;
            if width == 0.0 || height == 0.0 {
                return Err(anyhow::anyhow!(
                    "Invalid resolution detected from capture device"
                ));
            }

            println!(
                "✅ Frame capture initialized using: {}, resolution: {}x{}, FPS: 90",
                source, width, height
            );

            Ok(Self {
                cap,
                source,
                width,
                height,
                max_retries,
                retry_delay,
            })
        }

        pub fn capture_frame(&mut self) -> Result<core::Mat> {
            for attempt in 1..=self.max_retries {
                let mut frame = core::Mat::default();
                match self.cap.read(&mut frame) {
                    Ok(_) if !frame.empty() => {
                        let frame_width = frame.cols() as f32;
                        let frame_height = frame.rows() as f32;
                        // Accept frames that match the device resolution (no strict validation)
                        if frame_width == self.width && frame_height == self.height {
                            return Ok(frame);
                        } else {
                            println!(
                                "⚠️  Frame size {}x{} does not match device {}x{}, attempt {}/{}",
                                frame_width,
                                frame_height,
                                self.width,
                                self.height,
                                attempt,
                                self.max_retries
                            );
                        }
                    }
                    Ok(_) => {
                        println!(
                            "⚠️  Empty frame captured from {}, attempt {}/{}",
                            self.source, attempt, self.max_retries
                        );
                    }
                    Err(e) => {
                        println!(
                            "❌ Frame capture error from {}: {}, attempt {}/{}",
                            self.source, e, attempt, self.max_retries
                        );
                    }
                }
                std::thread::sleep(self.retry_delay);
            }
            Err(anyhow::anyhow!(
                "Failed to capture frame from {} after {} attempts",
                self.source,
                self.max_retries
            ))
        }

        pub fn get_source(&self) -> &str {
            &self.source
        }

        pub fn get_resolution(&self) -> (f32, f32) {
            (self.width, self.height)
        }
    }

    pub struct YoloDetector {
        session: Session<'static>,
        class_names: Vec<String>,
        conf_threshold: f32,
        nms_threshold: f32,
    }

    impl YoloDetector {
        pub fn new(model_path: String) -> Result<Self> {
            let class_names = coco80();
            let session = ORT_ENV
                .new_session_builder()?
                .with_optimization_level(onnxruntime::GraphOptimizationLevel::All)?
                .with_number_threads(4)?
                .with_model_from_file(model_path)?;
            Ok(Self {
                session,
                class_names,
                conf_threshold: 0.25,
                nms_threshold: 0.45,
            })
        }

        pub fn detect(&mut self, frame: &core::Mat) -> Result<Vec<Detection>> {
            let input_size = 640;
            let (input_tensor, lb) = letterbox_bgr_to_rgb_nchw(frame, input_size)?;

            let frame_width = frame.cols() as f32;
            let frame_height = frame.rows() as f32;

            let output_view = {
                let outputs = self.session.run(vec![input_tensor])?;
                outputs[0].view().to_owned()
            };

            let mut dets =
                self.parse_yolo_outputs(&output_view.view(), input_size as f32, input_size as f32)?;

            for d in dets.iter_mut() {
                let cx = d.bbox.x + d.bbox.width * 0.5;
                let cy = d.bbox.y + d.bbox.height * 0.5;
                let cx0 = (cx - lb.pad_x) / lb.scale;
                let cy0 = (cy - lb.pad_y) / lb.scale;
                let w0 = d.bbox.width / lb.scale;
                let h0 = d.bbox.height / lb.scale;
                d.bbox.x = cx0 - w0 * 0.5;
                d.bbox.y = cy0 - h0 * 0.5;
                d.bbox.width = w0;
                d.bbox.height = h0;
                d.bbox.x = d.bbox.x.max(0.0).min(frame_width - 1.0);
                d.bbox.y = d.bbox.y.max(0.0).min(frame_height - 1.0);
            }

            if dets.is_empty() {
                println!("🔍 YOLO produced 0 mapped detections this frame");
            } else {
                let labels: Vec<_> = dets.iter().map(|d| d.class_name.as_str()).collect();
                println!("✅ YOLO mapped detections: {}", labels.join(", "));
            }
            Ok(self.apply_nms(dets))
        }

        fn parse_yolo_outputs(
            &self,
            output: &onnxruntime::ndarray::ArrayViewD<f32>,
            img_width: f32,
            img_height: f32,
        ) -> Result<Vec<Detection>> {
            use ndarray::{Axis, Ix3};
            let mut out = Vec::new();

            if output.ndim() != 3 || output.shape()[0] != 1 {
                return Err(anyhow::anyhow!(
                    "Unexpected YOLO output shape: {:?}",
                    output.shape()
                ));
            }

            let need_transpose = {
                let s1 = output.shape()[1];
                let s2 = output.shape()[2];
                (s1 == 84 || s1 == 85) && !(s2 == 84 || s2 == 85)
            };

            let v = output.to_owned();
            let arr3 = v.into_dimensionality::<Ix3>().map_err(|_| {
                anyhow::anyhow!("Expected 3D YOLO output, got {:?}", output.shape())
            })?;

            let arr3 = if need_transpose {
                arr3.permuted_axes([0, 2, 1])
            } else {
                arr3
            };

            let pred = arr3.index_axis(Axis(0), 0);
            let n_boxes = pred.dim().0;
            let c_dim = pred.dim().1;

            let has_objness =
                c_dim == 85 || (c_dim > 10 && c_dim - 4 - self.class_names.len() == 1);
            let base = 4usize;
            let class_start = if has_objness { base + 1 } else { base };
            let num_classes = c_dim - class_start;

            let use_labels: Vec<String> = if num_classes == self.class_names.len() {
                self.class_names.clone()
            } else {
                (0..num_classes).map(|i| format!("cls_{}", i)).collect()
            };

            let input_size = 640.0;
            let sx = img_width / input_size;
            let sy = img_height / input_size;

            let mut candidates: Vec<(f32, f32, f32, f32, usize, f32)> = Vec::new();
            for i in 0..n_boxes {
                let cx = pred[[i, 0]];
                let cy = pred[[i, 1]];
                let w = pred[[i, 2]];
                let h = pred[[i, 3]];
                let obj = if has_objness { pred[[i, base]] } else { 1.0 };

                let mut best_cls = 0usize;
                let mut best_score = 0.0f32;
                for c in 0..num_classes {
                    let s = pred[[i, class_start + c]];
                    let sc = s * obj;
                    if sc > best_score {
                        best_score = sc;
                        best_cls = c;
                    }
                }
                if best_score < self.conf_threshold {
                    continue;
                }

                let cx_img = cx * sx;
                let cy_img = cy * sy;
                let w_img = w * sx;
                let h_img = h * sy;

                let x = cx_img - w_img * 0.5;
                let y = cy_img - h_img * 0.5;

                candidates.push((x, y, w_img, h_img, best_cls, best_score));
            }

            for (x, y, w, h, cls, score) in candidates {
                let raw = &use_labels[cls];
                if let Some(mapped) = map_coco_to_game(raw.as_str()) {
                    out.push(Detection {
                        class_name: mapped.to_string(),
                        confidence: score,
                        bbox: BoundingBox {
                            x,
                            y,
                            width: w,
                            height: h,
                        },
                    });
                }
            }

            Ok(out)
        }

        fn apply_nms(&self, mut detections: Vec<Detection>) -> Vec<Detection> {
            detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
            let mut keep = Vec::new();
            let mut suppressed = vec![false; detections.len()];

            for i in 0..detections.len() {
                if suppressed[i] {
                    continue;
                }

                keep.push(detections[i].clone());

                for j in (i + 1)..detections.len() {
                    if suppressed[j] {
                        continue;
                    }

                    let iou = self.calculate_iou(&detections[i].bbox, &detections[j].bbox);
                    if iou > self.nms_threshold {
                        suppressed[j] = true;
                    }
                }
            }

            keep
        }

        fn calculate_iou(&self, box1: &BoundingBox, box2: &BoundingBox) -> f32 {
            let x1 = box1.x.max(box2.x);
            let y1 = box1.y.max(box2.y);
            let x2 = (box1.x + box1.width).min(box2.x + box2.width);
            let y2 = (box1.y + box1.height).min(box2.y + box2.height);

            if x2 <= x1 || y2 <= y1 {
                return 0.0;
            }

            let intersection = (x2 - x1) * (y2 - y1);
            let area1 = box1.width * box1.height;
            let area2 = box2.width * box2.height;
            let union = area1 + area2 - intersection;

            intersection / union
        }
    }
}

mod control {
    use super::*;

    #[derive(Debug, Clone)]
    pub enum Action {
        Jump,
        Slide,
        MoveLeft,
        MoveRight,
        None,
    }

    pub struct AdbController {
        device_id: Option<String>,
    }

    impl AdbController {
        pub fn new(device_id: Option<String>) -> Self {
            Self { device_id }
        }

        pub async fn execute_action(&self, action: Action) -> Result<()> {
            match action {
                Action::Jump => self.swipe_up().await,
                Action::Slide => self.swipe_down().await,
                Action::MoveLeft => self.swipe_left().await,
                Action::MoveRight => self.swipe_right().await,
                Action::None => Ok(()),
            }
        }

        async fn swipe_up(&self) -> Result<()> {
            self.execute_swipe(540, 1750, 540, 750, 150).await
        }

        async fn swipe_down(&self) -> Result<()> {
            self.execute_swipe(540, 750, 540, 1750, 150).await
        }

        async fn swipe_left(&self) -> Result<()> {
            self.execute_swipe(800, 1170, 300, 1170, 150).await
        }

        async fn swipe_right(&self) -> Result<()> {
            self.execute_swipe(300, 1170, 800, 1170, 150).await
        }

        async fn execute_swipe(
            &self,
            x1: u32,
            y1: u32,
            x2: u32,
            y2: u32,
            duration: u32,
        ) -> Result<()> {
            let mut cmd = Command::new("adb");

            if let Some(ref device) = self.device_id {
                cmd.args(["-s", device]);
            }

            cmd.args([
                "shell",
                "input",
                "swipe",
                &x1.to_string(),
                &y1.to_string(),
                &x2.to_string(),
                &y2.to_string(),
                &duration.to_string(),
            ]);

            let output = cmd.output().await?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(anyhow::anyhow!("ADB command failed: {}", stderr));
            }

            Ok(())
        }
    }
}

mod decision {
    use super::*;

    pub struct GameDecisionEngine {
        screen_width: f32,
        screen_height: f32,
        lane_width: f32,
        player_lane: i32,
        last_action_time: Instant,
        action_cooldown: Duration,
        threat_conf_threshold: f32,
    }

    impl GameDecisionEngine {
        pub fn new(screen_width: f32, screen_height: f32) -> Self {
            Self {
                screen_width,
                screen_height,
                lane_width: screen_width / 3.0,
                player_lane: 1,
                last_action_time: Instant::now(),
                action_cooldown: Duration::from_millis(200),
                threat_conf_threshold: 0.3,
            }
        }

        pub fn decide_action(&mut self, detections: &[vision::Detection]) -> control::Action {
            if self.last_action_time.elapsed() < self.action_cooldown {
                return control::Action::None;
            }

            if let Some(player) = detections.iter().find(|d| d.class_name == "player") {
                self.update_player_lane(&player.bbox);
            }

            if let Some(action) = self.avoid_immediate_threats(detections) {
                self.last_action_time = Instant::now();
                self.update_player_lane_prediction(&action);
                return action;
            }

            control::Action::None
        }

        fn update_player_lane(&mut self, player_bbox: &vision::BoundingBox) {
            let center_x = player_bbox.x + player_bbox.width / 2.0;
            self.player_lane = if center_x < self.lane_width {
                0
            } else if center_x < self.lane_width * 2.0 {
                1
            } else {
                2
            };
        }

        fn update_player_lane_prediction(&mut self, action: &control::Action) {
            match action {
                control::Action::MoveLeft => {
                    if self.player_lane > 0 {
                        self.player_lane -= 1;
                    }
                }
                control::Action::MoveRight => {
                    if self.player_lane < 2 {
                        self.player_lane += 1;
                    }
                }
                _ => {}
            }
        }

        fn avoid_immediate_threats(
            &self,
            detections: &[vision::Detection],
        ) -> Option<control::Action> {
            let threat_distance = self.screen_height * 0.35;

            let mut sorted_detections = detections.to_vec();
            sorted_detections.sort_by(|a, b| {
                let dist_a = self.screen_height - (a.bbox.y + a.bbox.height);
                let dist_b = self.screen_height - (b.bbox.y + b.bbox.height);
                dist_a
                    .partial_cmp(&dist_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for detection in &sorted_detections {
                if detection.confidence < self.threat_conf_threshold {
                    continue;
                }

                let distance_from_bottom =
                    self.screen_height - (detection.bbox.y + detection.bbox.height);
                if distance_from_bottom > threat_distance {
                    continue;
                }

                let object_lane = self.get_object_lane(&detection.bbox);
                if object_lane != self.player_lane {
                    continue;
                }

                match detection.class_name.as_str() {
                    "train_blocking" => return self.choose_safe_lane(detections, threat_distance),
                    "barrier_overhead" => return Some(control::Action::Slide),
                    "barrier_ground" => return Some(control::Action::Jump),
                    _ => {}
                }
            }

            None
        }

        fn choose_safe_lane(
            &self,
            detections: &[vision::Detection],
            threat_distance: f32,
        ) -> Option<control::Action> {
            if self.player_lane == 0 {
                return Some(control::Action::MoveRight);
            } else if self.player_lane == 2 {
                return Some(control::Action::MoveLeft);
            } else {
                let left_clear = self.is_lane_safe(detections, 0, threat_distance);
                let right_clear = self.is_lane_safe(detections, 2, threat_distance);

                if left_clear && !right_clear {
                    return Some(control::Action::MoveLeft);
                } else if right_clear && !left_clear {
                    return Some(control::Action::MoveRight);
                } else if left_clear && right_clear {
                    return Some(control::Action::MoveLeft);
                }
            }
            None
        }

        fn get_object_lane(&self, bbox: &vision::BoundingBox) -> i32 {
            let center_x = bbox.x + bbox.width / 2.0;
            if center_x < self.lane_width {
                0
            } else if center_x < self.lane_width * 2.0 {
                1
            } else {
                2
            }
        }

        fn is_lane_safe(
            &self,
            detections: &[vision::Detection],
            lane: i32,
            check_distance: f32,
        ) -> bool {
            for detection in detections {
                if detection.confidence < self.threat_conf_threshold {
                    continue;
                }

                let distance_from_bottom =
                    self.screen_height - (detection.bbox.y + detection.bbox.height);
                if distance_from_bottom > check_distance {
                    continue;
                }

                let object_lane = self.get_object_lane(&detection.bbox);
                if object_lane == lane {
                    match detection.class_name.as_str() {
                        "train_blocking" | "barrier_overhead" | "barrier_ground" => return false,
                        _ => {}
                    }
                }
            }
            true
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎮 Subway Surfers Bot Starting...");

    let scrcpy_cmd = Command::new("scrcpy")
        .args([
            "--v4l2-sink=/dev/video2",
            "--fullscreen",
            "--max-fps=90",
            "--no-audio",
            "--no-control",
            "--no-playback",
        ])
        .spawn();

    if let Err(e) = scrcpy_cmd {
        eprintln!(
            "❌ Failed to start scrcpy: {}. Ensure scrcpy is installed and device is connected.",
            e
        );
        return Err(anyhow::anyhow!("Scrcpy startup failed"));
    }

    sleep(Duration::from_millis(2000)).await;

    apply_profile("subway_surfers");
    println!("⚙️  Applied Subway Surfers detection profile");

    let mut frame_capture = vision::FrameCapture::new().map_err(|e| {
        anyhow::anyhow!(
            "Failed to initialize frame capture: {}. Ensure scrcpy is running.",
            e
        )
    })?;

    let (screen_width, screen_height) = frame_capture.get_resolution();
    let mut yolo_detector = vision::YoloDetector::new("models/subway_surfers.onnx".to_string())
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to load YOLO model: {}. Make sure the model file exists.",
                e
            )
        })?;

    let adb_controller = control::AdbController::new(None);
    let mut decision_engine = decision::GameDecisionEngine::new(screen_width, screen_height);

    println!("✅ All components initialized successfully");
    println!("🚀 Starting game automation loop...");

    let mut frame_count = 0;
    let mut total_detections = 0;
    let start_time = Instant::now();
    let mut last_successful_frame = Instant::now();
    let max_stall_duration = Duration::from_secs(5);

    loop {
        let loop_start = Instant::now();

        let frame = match frame_capture.capture_frame() {
            Ok(frame) => {
                last_successful_frame = Instant::now();
                frame
            }
            Err(e) => {
                eprintln!("❌ Frame capture error: {}", e);
                if last_successful_frame.elapsed() > max_stall_duration {
                    return Err(anyhow::anyhow!("Frame capture stalled for too long"));
                }
                sleep(Duration::from_millis(33)).await;
                continue;
            }
        };

        let detections = match yolo_detector.detect(&frame) {
            Ok(detections) => detections,
            Err(e) => {
                eprintln!("❌ Detection error: {}", e);
                sleep(Duration::from_millis(33)).await;
                continue;
            }
        };

        total_detections += detections.len();

        let action = decision_engine.decide_action(&detections);

        if let Err(e) = adb_controller.execute_action(action.clone()).await {
            eprintln!("❌ ADB action error: {}", e);
        } else if !matches!(action, control::Action::None) {
            println!(
                "🎯 Executed action: {:?} (based on {} detections: {})",
                action,
                detections.len(),
                detections
                    .iter()
                    .map(|d| d.class_name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        frame_count += 1;

        if frame_count % 100 == 0 {
            let elapsed = start_time.elapsed();
            let fps = frame_count as f64 / elapsed.as_secs_f64();
            let avg_detections = total_detections as f64 / frame_count as f64;

            println!(
                "📊 Stats: {} frames, {:.1} FPS, {:.1} avg detections/frame",
                frame_count, fps, avg_detections
            );

            with_config(|cfg| {
                println!(
                    "⚙️  Config: latency_budget={}ms, crop_top={:.1}%",
                    cfg.latency_budget_ms,
                    cfg.crop_top_frac * 100.0
                );
            });
        }

        let loop_duration = loop_start.elapsed();
        let target_frame_time = Duration::from_millis(11);
        if loop_duration < target_frame_time {
            sleep(target_frame_time - loop_duration).await;
        }
    }
}
