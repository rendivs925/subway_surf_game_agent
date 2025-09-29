use anyhow::Result;
use opencv::{core, imgproc, prelude::*, videoio};
use onnxruntime::{environment::Environment, session::Session, GraphOptimizationLevel, LoggingLevel};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tokio::process::Command;
use tokio::time::sleep;

static ORT_ENV: Lazy<Environment> = Lazy::new(|| {
    Environment::builder()
        .with_name("subway_surfers_detector")
        .with_log_level(LoggingLevel::Warning)
        .build()
        .expect("Failed to create ONNX Runtime environment")
});

// Vision module - handles frame capture and YOLO detection
mod vision {
    use super::*;

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
    }

    impl FrameCapture {
        pub fn new() -> Result<Self> {
            // Try to open scrcpy v4l2loopback device first (/dev/video2)
            let mut cap = videoio::VideoCapture::new(2, videoio::CAP_V4L2)?;
            let mut source = "/dev/video2 (scrcpy v4l2loopback)".to_string();

            if !cap.is_opened()? {
                println!("⚠️  Failed to open /dev/video2, trying fallback to /dev/video0...");

                // Fallback to default camera device
                cap = videoio::VideoCapture::new(0, videoio::CAP_V4L2)?;
                source = "/dev/video0 (fallback)".to_string();

                if !cap.is_opened()? {
                    return Err(anyhow::anyhow!(
                        "Failed to open both /dev/video2 and /dev/video0. \
                        Make sure scrcpy is running with --v4l2-sink=/dev/video2 or a camera is connected."
                    ));
                }
            }

            // Configure capture properties
            cap.set(videoio::CAP_PROP_FRAME_WIDTH, 720.0)?;
            cap.set(videoio::CAP_PROP_FRAME_HEIGHT, 1280.0)?;
            cap.set(videoio::CAP_PROP_FPS, 30.0)?;
            cap.set(videoio::CAP_PROP_BUFFERSIZE, 1.0)?; // Reduce buffer for lower latency

            println!("✅ Frame capture initialized using: {}", source);

            Ok(Self { cap, source })
        }

        pub fn capture_frame(&mut self) -> Result<core::Mat> {
            let mut frame = core::Mat::default();
            self.cap.read(&mut frame)?;

            if frame.empty() {
                return Err(anyhow::anyhow!(
                    "Empty frame captured from {}. Check if scrcpy is running with --v4l2-sink=/dev/video2",
                    self.source
                ));
            }

            Ok(frame)
        }

        pub fn get_source(&self) -> &str {
            &self.source
        }
    }

    pub struct YoloDetector {
        session: Session<'static>,
        class_names: Vec<String>,
    }

    impl YoloDetector {
        pub fn new(model_path: &str) -> Result<Self> {
            let class_names = vec![
                "player".to_string(),
                "coin".to_string(),
                "train_blocking".to_string(),
                "train_jumpable".to_string(),
                "train_free".to_string(),
                "barrier_overhead".to_string(),
                "barrier_ground".to_string(),
            ];

            let session = ORT_ENV
                .new_session_builder()?
                .with_optimization_level(GraphOptimizationLevel::All)?
                .with_number_threads(4)?
                .with_model_from_file(model_path.to_string())?;

            Ok(Self {
                session,
                class_names,
            })
        }

        pub fn detect(&mut self, frame: &core::Mat) -> Result<Vec<Detection>> {
            // Preprocess frame for YOLO
            let input_size = 640;
            let mut resized = core::Mat::default();
            imgproc::resize(
                frame,
                &mut resized,
                core::Size::new(input_size, input_size),
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;

            // Convert BGR to RGB and normalize
            let mut rgb = core::Mat::default();
            imgproc::cvt_color(&resized, &mut rgb, imgproc::COLOR_BGR2RGB, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;

            // Convert to NCHW format for ONNX
            let input_array = self.mat_to_array(&rgb)?;
            let input_tensor = input_array.into_owned();

            // Extract frame dimensions before running inference
            let frame_width = frame.cols() as f32;
            let frame_height = frame.rows() as f32;

            // Run inference and extract output data
            let output_view = {
                let outputs = self.session.run(vec![input_tensor])?;
                outputs[0].view().to_owned()
            };

            // Parse YOLO outputs
            self.parse_yolo_outputs(&output_view.view(), frame_width, frame_height)
        }

        fn mat_to_array(&self, mat: &core::Mat) -> Result<ndarray::Array4<f32>> {
            let size = mat.size()?;
            let mut array = ndarray::Array4::<f32>::zeros((1, 3, size.height as usize, size.width as usize));

            unsafe {
                let data = mat.ptr(0)? as *const u8;
                for y in 0..size.height as usize {
                    for x in 0..size.width as usize {
                        let pixel_idx = (y * size.width as usize + x) * 3;
                        let r = *data.add(pixel_idx + 2) as f32 / 255.0;
                        let g = *data.add(pixel_idx + 1) as f32 / 255.0;
                        let b = *data.add(pixel_idx) as f32 / 255.0;

                        array[[0, 0, y, x]] = r;
                        array[[0, 1, y, x]] = g;
                        array[[0, 2, y, x]] = b;
                    }
                }
            }

            Ok(array)
        }

        fn parse_yolo_outputs(&self, output: &onnxruntime::ndarray::ArrayViewD<f32>, img_width: f32, img_height: f32) -> Result<Vec<Detection>> {
            let mut detections = Vec::new();
            let confidence_threshold = 0.5;
            let nms_threshold = 0.4;

            // Assuming YOLOv5/v8 output format: [batch, 25200, 85] where 85 = 4 (bbox) + 1 (objectness) + 80 (classes)
            // For our 7 classes: [batch, 25200, 12] where 12 = 4 (bbox) + 1 (objectness) + 7 (classes)

            if output.ndim() != 3 {
                return Err(anyhow::anyhow!("Unexpected output dimensions"));
            }

            let shape = output.shape();
            let num_detections = shape[1];
            let _num_features = shape[2];

            for i in 0..num_detections {
                let objectness = output[[0, i, 4]];
                if objectness < confidence_threshold {
                    continue;
                }

                // Find best class
                let mut best_class_idx = 0;
                let mut best_class_score = 0.0;
                for j in 0..self.class_names.len() {
                    let class_score = output[[0, i, 5 + j]];
                    if class_score > best_class_score {
                        best_class_score = class_score;
                        best_class_idx = j;
                    }
                }

                let final_confidence = objectness * best_class_score;
                if final_confidence < confidence_threshold {
                    continue;
                }

                // Parse bounding box (center_x, center_y, width, height)
                let center_x = output[[0, i, 0]] * img_width / 640.0;
                let center_y = output[[0, i, 1]] * img_height / 640.0;
                let width = output[[0, i, 2]] * img_width / 640.0;
                let height = output[[0, i, 3]] * img_height / 640.0;

                let x = center_x - width / 2.0;
                let y = center_y - height / 2.0;

                detections.push(Detection {
                    class_name: self.class_names[best_class_idx].clone(),
                    confidence: final_confidence,
                    bbox: BoundingBox { x, y, width, height },
                });
            }

            // Apply NMS
            Ok(self.apply_nms(detections, nms_threshold))
        }

        fn apply_nms(&self, mut detections: Vec<Detection>, threshold: f32) -> Vec<Detection> {
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
                    if iou > threshold {
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

// Control module - handles ADB commands
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
            self.execute_swipe(500, 1500, 500, 500, 150).await
        }

        async fn swipe_down(&self) -> Result<()> {
            self.execute_swipe(500, 500, 500, 1500, 150).await
        }

        async fn swipe_left(&self) -> Result<()> {
            self.execute_swipe(800, 1000, 200, 1000, 150).await
        }

        async fn swipe_right(&self) -> Result<()> {
            self.execute_swipe(200, 1000, 800, 1000, 150).await
        }

        async fn execute_swipe(&self, x1: u32, y1: u32, x2: u32, y2: u32, duration: u32) -> Result<()> {
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

// Decision module - game logic
mod decision {
    use super::*;

    pub struct GameDecisionEngine {
        screen_width: f32,
        screen_height: f32,
        lane_width: f32,
        player_lane: i32, // 0=left, 1=center, 2=right
        last_action_time: Instant,
        action_cooldown: Duration,
    }

    impl GameDecisionEngine {
        pub fn new(screen_width: f32, screen_height: f32) -> Self {
            Self {
                screen_width,
                screen_height,
                lane_width: screen_width / 3.0,
                player_lane: 1, // Start in center
                last_action_time: Instant::now(),
                action_cooldown: Duration::from_millis(300),
            }
        }

        pub fn decide_action(&mut self, detections: &[vision::Detection]) -> control::Action {
            // Cooldown check
            if self.last_action_time.elapsed() < self.action_cooldown {
                return control::Action::None;
            }

            // Get player position to update current lane
            if let Some(player) = detections.iter().find(|d| d.class_name == "player") {
                self.update_player_lane(&player.bbox);
            }

            // Priority 1: Avoid immediate threats
            if let Some(action) = self.avoid_immediate_threats(detections) {
                self.last_action_time = Instant::now();
                self.update_player_lane_prediction(&action);
                return action;
            }

            // Priority 2: Collect coins if safe
            if let Some(action) = self.collect_coins_safely(detections) {
                self.last_action_time = Instant::now();
                self.update_player_lane_prediction(&action);
                return action;
            }

            control::Action::None
        }

        fn update_player_lane(&mut self, player_bbox: &vision::BoundingBox) {
            let center_x = player_bbox.x + player_bbox.width / 2.0;

            if center_x < self.lane_width {
                self.player_lane = 0; // Left
            } else if center_x < self.lane_width * 2.0 {
                self.player_lane = 1; // Center
            } else {
                self.player_lane = 2; // Right
            }
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

        fn avoid_immediate_threats(&self, detections: &[vision::Detection]) -> Option<control::Action> {
            let threat_distance = self.screen_height * 0.4; // Consider threats in bottom 40% of screen

            for detection in detections {
                let distance_from_bottom = self.screen_height - (detection.bbox.y + detection.bbox.height);

                if distance_from_bottom > threat_distance {
                    continue; // Too far away
                }

                let object_lane = self.get_object_lane(&detection.bbox);

                match detection.class_name.as_str() {
                    "train_blocking" => {
                        if object_lane == self.player_lane {
                            // Need to move to adjacent lane
                            if self.player_lane == 0 {
                                return Some(control::Action::MoveRight);
                            } else if self.player_lane == 2 {
                                return Some(control::Action::MoveLeft);
                            } else {
                                // In center, choose safer side
                                let left_clear = self.is_lane_safe(detections, 0, threat_distance);
                                let right_clear = self.is_lane_safe(detections, 2, threat_distance);

                                if left_clear && !right_clear {
                                    return Some(control::Action::MoveLeft);
                                } else if right_clear && !left_clear {
                                    return Some(control::Action::MoveRight);
                                } else if left_clear && right_clear {
                                    return Some(control::Action::MoveLeft); // Default to left
                                }
                            }
                        }
                    }
                    "train_jumpable" => {
                        if object_lane == self.player_lane {
                            return Some(control::Action::Jump);
                        }
                    }
                    "barrier_overhead" => {
                        if object_lane == self.player_lane {
                            return Some(control::Action::Slide);
                        }
                    }
                    "barrier_ground" => {
                        if object_lane == self.player_lane {
                            return Some(control::Action::Jump);
                        }
                    }
                    _ => {}
                }
            }

            None
        }

        fn collect_coins_safely(&self, detections: &[vision::Detection]) -> Option<control::Action> {
            let coin_distance = self.screen_height * 0.6; // Consider coins in bottom 60% of screen

            for detection in detections {
                if detection.class_name != "coin" {
                    continue;
                }

                let distance_from_bottom = self.screen_height - (detection.bbox.y + detection.bbox.height);

                if distance_from_bottom > coin_distance {
                    continue; // Too far away
                }

                let coin_lane = self.get_object_lane(&detection.bbox);

                if coin_lane != self.player_lane {
                    // Check if target lane is safe
                    if self.is_lane_safe(detections, coin_lane, coin_distance) {
                        if coin_lane < self.player_lane {
                            return Some(control::Action::MoveLeft);
                        } else {
                            return Some(control::Action::MoveRight);
                        }
                    }
                }
            }

            None
        }

        fn get_object_lane(&self, bbox: &vision::BoundingBox) -> i32 {
            let center_x = bbox.x + bbox.width / 2.0;

            if center_x < self.lane_width {
                0 // Left
            } else if center_x < self.lane_width * 2.0 {
                1 // Center
            } else {
                2 // Right
            }
        }

        fn is_lane_safe(&self, detections: &[vision::Detection], lane: i32, check_distance: f32) -> bool {
            for detection in detections {
                let distance_from_bottom = self.screen_height - (detection.bbox.y + detection.bbox.height);

                if distance_from_bottom > check_distance {
                    continue;
                }

                let object_lane = self.get_object_lane(&detection.bbox);

                if object_lane == lane {
                    match detection.class_name.as_str() {
                        "train_blocking" | "barrier_overhead" | "barrier_ground" => {
                            return false;
                        }
                        _ => {}
                    }
                }
            }

            true
        }
    }
}

// Main application
#[tokio::main]
async fn main() -> Result<()> {
    println!("🎮 Subway Surfers Bot Starting...");

    // Initialize components
    let mut frame_capture = vision::FrameCapture::new()
        .map_err(|e| anyhow::anyhow!("Failed to initialize frame capture: {}. Make sure scrcpy is running.", e))?;

    let mut yolo_detector = vision::YoloDetector::new("models/subway_surfers.onnx")
        .map_err(|e| anyhow::anyhow!("Failed to load YOLO model: {}. Make sure the model file exists.", e))?;

    let adb_controller = control::AdbController::new(None); // Auto-detect device

    let mut decision_engine = decision::GameDecisionEngine::new(720.0, 1280.0);

    println!("✅ All components initialized successfully");
    println!("🚀 Starting game automation loop...");

    let mut frame_count = 0;
    let mut total_detections = 0;
    let start_time = Instant::now();

    loop {
        let loop_start = Instant::now();

        // Capture frame
        let frame = match frame_capture.capture_frame() {
            Ok(frame) => frame,
            Err(e) => {
                eprintln!("❌ Frame capture error: {}", e);
                sleep(Duration::from_millis(33)).await; // ~30 FPS
                continue;
            }
        };

        // Run detection
        let detections = match yolo_detector.detect(&frame) {
            Ok(detections) => detections,
            Err(e) => {
                eprintln!("❌ Detection error: {}", e);
                sleep(Duration::from_millis(33)).await;
                continue;
            }
        };

        total_detections += detections.len();

        // Make decision
        let action = decision_engine.decide_action(&detections);

        // Execute action
        if let Err(e) = adb_controller.execute_action(action.clone()).await {
            eprintln!("❌ ADB action error: {}", e);
        } else if !matches!(action, control::Action::None) {
            println!("🎯 Executed action: {:?}", action);
        }

        frame_count += 1;

        // Print stats every 100 frames
        if frame_count % 100 == 0 {
            let elapsed = start_time.elapsed();
            let fps = frame_count as f64 / elapsed.as_secs_f64();
            let avg_detections = total_detections as f64 / frame_count as f64;

            println!(
                "📊 Stats: {} frames, {:.1} FPS, {:.1} avg detections/frame",
                frame_count, fps, avg_detections
            );
        }

        // Maintain target FPS (30 FPS = 33ms per frame)
        let loop_duration = loop_start.elapsed();
        let target_frame_time = Duration::from_millis(33);

        if loop_duration < target_frame_time {
            sleep(target_frame_time - loop_duration).await;
        }
    }
}