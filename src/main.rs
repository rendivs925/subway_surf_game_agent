use anyhow::Result;
use opencv::{core, imgcodecs, imgproc, prelude::*, videoio};
use std::time::{Duration, Instant};
use tokio::process::Command;
use tokio::time::sleep;

// ============================== Utilities ==============================

fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

fn roi_clone(frame: &core::Mat, rect: core::Rect) -> Result<core::Mat> {
    let r = safe_rect(frame, rect)?;
    let v = core::Mat::roi(frame, r)?;
    Ok(v.try_clone()?)
}

fn safe_rect(frame: &core::Mat, mut r: core::Rect) -> Result<core::Rect> {
    let w = frame.cols();
    let h = frame.rows();
    if r.x < 0 {
        r.x = 0;
    }
    if r.y < 0 {
        r.y = 0;
    }
    if r.x + r.width > w {
        r.width = (w - r.x).max(0);
    }
    if r.y + r.height > h {
        r.height = (h - r.y).max(0);
    }
    if r.width <= 0 || r.height <= 0 {
        return Err(anyhow::anyhow!("Invalid ROI after clamping"));
    }
    Ok(r)
}

struct ExponentialSmooth {
    alpha: f32,
    val: f32,
    inited: bool,
}
impl ExponentialSmooth {
    fn new(alpha: f32) -> Self {
        Self {
            alpha,
            val: 0.0,
            inited: false,
        }
    }
    fn step(&mut self, x: f32) -> f32 {
        if !self.inited {
            self.val = x;
            self.inited = true;
            return self.val;
        }
        self.val = self.alpha * x + (1.0 - self.alpha) * self.val;
        self.val
    }
}

// ============================== Capture ==============================

struct FrameCapture {
    cap: videoio::VideoCapture,
    width: i32,
    height: i32,
}
impl FrameCapture {
    pub fn new() -> Result<Self> {
        let cap = videoio::VideoCapture::new(2, videoio::CAP_V4L2)?;
        if !cap.is_opened()? {
            return Err(anyhow::anyhow!(
                "Failed to open /dev/video2. Run: scrcpy --v4l2-sink=/dev/video2 --no-audio --no-control --max-fps=60"
            ));
        }
        let width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
        let height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
        println!("✅ Capture: {}x{}", width, height);
        Ok(Self { cap, width, height })
    }
    pub fn grab(&mut self) -> Result<core::Mat> {
        let mut frame = core::Mat::default();
        self.cap.read(&mut frame)?;
        if frame.empty() {
            return Err(anyhow::anyhow!("Empty frame from scrcpy"));
        }
        Ok(frame)
    }
    pub fn size(&self) -> (i32, i32) {
        (self.width, self.height)
    }
}

// ============================== Vision Config ==============================

struct VisionConfig {
    fuel_roi: core::Rect,    // where fuel spawns (top-middle corridor)
    car_roi: core::Rect,     // car + wheels (bottom-center)
    gauge_roi: core::Rect,   // fuel gauge bar top-left
    ground_band: core::Rect, // bottom band for slope
}
impl VisionConfig {
    fn for_frame(w: i32, h: i32) -> Self {
        let fuel_roi = core::Rect::new(w / 3, 0, w / 3, h / 2);
        let car_roi = core::Rect::new(w / 3, h / 2, w / 3, h / 2);
        let gauge_w = (w as f32 * 0.18) as i32;
        let gauge_h = (h as f32 * 0.06) as i32;
        let gauge_roi = core::Rect::new(
            (w as f32 * 0.02) as i32,
            (h as f32 * 0.04) as i32,
            gauge_w,
            gauge_h,
        );
        let band_h = (h as f32 * 0.18) as i32;
        let ground_band = core::Rect::new(0, h - band_h, w, band_h);
        Self {
            fuel_roi,
            car_roi,
            gauge_roi,
            ground_band,
        }
    }
}

// ============================== Vision ==============================

fn detect_fuel_multiscale(roi_bgr: &core::Mat, fuel_template: &core::Mat) -> Result<bool> {
    // Pre-blur ROI to reduce aliasing
    let mut roi_blur = core::Mat::default();
    imgproc::gaussian_blur(
        roi_bgr,
        &mut roi_blur,
        core::Size::new(3, 3),
        0.0,
        0.0,
        core::BorderTypes::BORDER_REFLECT as i32,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // Try multiple template scales to accommodate distance/perspective
    let mut best = 0.0f64;
    for scale in [1.00, 0.90, 0.80] {
        let mut tpl = core::Mat::default();
        let sz = core::Size::new(
            (fuel_template.cols() as f64 * scale) as i32,
            (fuel_template.rows() as f64 * scale) as i32,
        );
        if sz.width < 8 || sz.height < 8 {
            continue;
        }
        imgproc::resize(fuel_template, &mut tpl, sz, 0.0, 0.0, imgproc::INTER_AREA)?;

        let mut result = core::Mat::default();
        imgproc::match_template(
            &roi_blur,
            &tpl,
            &mut result,
            imgproc::TM_CCOEFF_NORMED,
            &core::no_array(),
        )?;
        let mut min_v = 0.0;
        let mut max_v = 0.0;
        core::min_max_loc(
            &result,
            Some(&mut min_v),
            Some(&mut max_v),
            None,
            None,
            &core::no_array(),
        )?;
        if max_v > best {
            best = max_v;
        }
    }
    Ok(best > 0.86)
}

fn estimate_car_tilt_deg(car_bgr: &core::Mat) -> Result<f32> {
    // Use median of Hough line angles; ignore near-verticals.
    let mut gray = core::Mat::default();
    imgproc::cvt_color(
        car_bgr,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let mut edges = core::Mat::default();
    imgproc::canny(&gray, &mut edges, 60.0, 160.0, 3, false)?;

    let mut lines = opencv::types::VectorOfVec4i::new();
    imgproc::hough_lines_p(
        &edges,
        &mut lines,
        1.0,
        std::f64::consts::PI / 180.0,
        40,
        28.0,
        14.0,
    )?;

    let mut angs: Vec<f32> = Vec::new();
    for l in lines.iter() {
        let dx = (l[2] - l[0]) as f32;
        let dy = (l[3] - l[1]) as f32;
        if dx.abs() < 2.0 {
            continue;
        }
        let a = (dy / dx).atan().to_degrees();
        if a.abs() < 55.0 {
            angs.push(a);
        } // reject near-verticals (background)
    }
    if angs.len() < 4 {
        return Ok(0.0);
    } // insufficient evidence → neutral
    angs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(angs[angs.len() / 2])
}

fn estimate_ground_slope_deg(band_bgr: &core::Mat) -> Result<f32> {
    // Bottom band slope: average near-horizontal edges
    let mut gray = core::Mat::default();
    imgproc::cvt_color(
        band_bgr,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let mut edges = core::Mat::default();
    imgproc::canny(&gray, &mut edges, 40.0, 120.0, 3, false)?;
    let mut lines = opencv::types::VectorOfVec4i::new();
    imgproc::hough_lines_p(
        &edges,
        &mut lines,
        1.0,
        std::f64::consts::PI / 180.0,
        30,
        40.0,
        20.0,
    )?;

    let mut total = 0.0f32;
    let mut n = 0;
    for l in lines.iter() {
        let dx = (l[2] - l[0]) as f32;
        let dy = (l[3] - l[1]) as f32;
        if dx.abs() < 2.0 {
            continue;
        }
        let a = (dy / dx).atan().to_degrees();
        if a.abs() < 35.0 {
            total += a;
            n += 1;
        } // near-horizontal = ground
    }
    Ok(if n > 0 { total / n as f32 } else { 0.0 })
}

fn estimate_fuel_level_smooth(gauge_bgr: &core::Mat, ema: &mut ExponentialSmooth) -> Result<f32> {
    // Green-ish HSV; wide range + morphological close + EMA smoothing
    let mut hsv = core::Mat::default();
    imgproc::cvt_color(
        gauge_bgr,
        &mut hsv,
        imgproc::COLOR_BGR2HSV,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut m1 = core::Mat::default();
    let mut m2 = core::Mat::default();
    core::in_range(
        &hsv,
        &core::Scalar::new(35.0, 40.0, 60.0, 0.0),
        &core::Scalar::new(85.0, 255.0, 255.0, 0.0),
        &mut m1,
    )?;
    core::in_range(
        &hsv,
        &core::Scalar::new(25.0, 40.0, 60.0, 0.0),
        &core::Scalar::new(35.0, 255.0, 255.0, 0.0),
        &mut m2,
    )?;

    let mut mask = core::Mat::default();
    core::bitwise_or(&m1, &m2, &mut mask, &core::no_array())?;

    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        core::Size::new(3, 3),
        core::Point::new(-1, -1),
    )?;
    let mut mask_closed = core::Mat::default();
    imgproc::morphology_ex(
        &mask,
        &mut mask_closed,
        imgproc::MORPH_CLOSE,
        &kernel,
        core::Point::new(-1, -1),
        1,
        core::BorderTypes::BORDER_CONSTANT as i32,
        core::Scalar::new(0.0, 0.0, 0.0, 0.0),
    )?;
    let mask = mask_closed;

    let total = (mask.rows() * mask.cols()) as f32;
    if total <= 0.0 {
        return Ok(ema.step(0.0));
    }
    let green = core::count_non_zero(&mask)? as f32;
    Ok(ema.step(clamp(green / total, 0.0, 1.0)))
}

struct AirborneDetector {
    last_contact: Instant,
    grace_ms: u64,
    prev_wheel_band_gray: Option<core::Mat>,
}
impl AirborneDetector {
    fn new(grace_ms: u64) -> Self {
        Self {
            last_contact: Instant::now(),
            grace_ms,
            prev_wheel_band_gray: None,
        }
    }

    fn detect(&mut self, car_bgr: &core::Mat) -> Result<bool> {
        // Look at a thin band at the very bottom of the car ROI
        let h = car_bgr.rows();
        let band_h = (h as f32 * 0.22) as i32;
        let band = roi_clone(
            car_bgr,
            core::Rect::new(0, h - band_h, car_bgr.cols(), band_h),
        )?;

        let mut gray = core::Mat::default();
        imgproc::cvt_color(
            &band,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        let mut edges = core::Mat::default();
        imgproc::canny(&gray, &mut edges, 60.0, 160.0, 3, false)?;
        let edge_ratio =
            (core::count_non_zero(&edges)? as f32) / ((edges.rows() * edges.cols()) as f32 + 1e-5);

        // Frame diff on band
        let diff_ratio = if let Some(prev) = &self.prev_wheel_band_gray {
            let mut diff = core::Mat::default();
            core::absdiff(&gray, prev, &mut diff)?;
            let mut th = core::Mat::default();
            imgproc::threshold(&diff, &mut th, 15.0, 255.0, imgproc::THRESH_BINARY)?;
            (core::count_non_zero(&th)? as f32) / ((th.rows() * th.cols()) as f32 + 1e-5)
        } else {
            0.1
        };

        self.prev_wheel_band_gray = Some(gray);

        let wheels_contact = edge_ratio > 0.03 || diff_ratio > 0.04;
        if wheels_contact {
            self.last_contact = Instant::now();
        }

        let airborne = (Instant::now() - self.last_contact).as_millis() as u64 > self.grace_ms;
        Ok(airborne)
    }
}

// ============================== Touch / ADB ==============================

struct TouchPoints {
    gas: (i32, i32),
    brake: (i32, i32),
}
impl TouchPoints {
    fn for_frame(w: i32, h: i32) -> Self {
        // bottom corners with margins
        let y = (h as f32 * 0.88) as i32;
        let gas_x = (w as f32 * 0.90) as i32;
        let brake_x = (w as f32 * 0.10) as i32;
        Self {
            gas: (gas_x, y),
            brake: (brake_x, y),
        }
    }
}

struct AdbController {
    device_id: Option<String>,
}
impl AdbController {
    fn new(device_id: Option<String>) -> Self {
        Self { device_id }
    }

    async fn long_press(&self, x: i32, y: i32, ms: i32) -> Result<()> {
        let mut cmd = Command::new("adb");
        if let Some(id) = &self.device_id {
            cmd.args(["-s", id]);
        }
        let out = cmd
            .args([
                "shell",
                "input",
                "swipe",
                &x.to_string(),
                &y.to_string(),
                &x.to_string(),
                &y.to_string(),
                &ms.to_string(),
            ])
            .output()
            .await?;
        if !out.status.success() {
            return Err(anyhow::anyhow!(
                "ADB swipe failed: {}",
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        Ok(())
    }
}

// ============================== Control Policy ==============================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Drive,
    Airborne,
    Recover,
}

struct PID {
    kp: f32,
    ki: f32,
    kd: f32,
    integ: f32,
    prev_err: f32,
    inited: bool,
}
impl PID {
    fn new(kp: f32, ki: f32, kd: f32) -> Self {
        Self {
            kp,
            ki,
            kd,
            integ: 0.0,
            prev_err: 0.0,
            inited: false,
        }
    }
    fn step(&mut self, err: f32, dt: f32) -> f32 {
        self.integ += err * dt;
        let deriv = if self.inited {
            (err - self.prev_err) / dt.max(1e-3)
        } else {
            0.0
        };
        self.prev_err = err;
        self.inited = true;
        self.kp * err + self.ki * self.integ + self.kd * deriv
    }
}

struct DecisionEngine {
    phase: Phase,
    last_phase_change: Instant,
    tilt_pid: PID,
    target_tilt: f32,
    tilt_limit: f32,
    throttle_smooth: ExponentialSmooth,
}
impl DecisionEngine {
    fn new() -> Self {
        Self {
            phase: Phase::Drive,
            last_phase_change: Instant::now(),
            tilt_pid: PID::new(0.9, 0.0, 0.28),
            target_tilt: 0.0,
            tilt_limit: 22.0,
            throttle_smooth: ExponentialSmooth::new(0.35),
        }
    }

    fn update_phase(&mut self, airborne: bool, tilt: f32) {
        match self.phase {
            Phase::Drive => {
                if airborne {
                    self.phase = Phase::Airborne;
                    self.last_phase_change = Instant::now();
                } else if tilt.abs() > self.tilt_limit {
                    self.phase = Phase::Recover;
                    self.last_phase_change = Instant::now();
                }
            }
            Phase::Airborne => {
                if !airborne {
                    self.phase = Phase::Drive;
                    self.last_phase_change = Instant::now();
                }
            }
            Phase::Recover => {
                if tilt.abs() < self.tilt_limit * 0.6 {
                    self.phase = Phase::Drive;
                    self.last_phase_change = Instant::now();
                }
            }
        }
    }

    /// Returns throttle in [-1..1] (negative = brake, positive = gas)
    fn decide(&mut self, tilt: f32, slope: f32, fuel_level: f32, fuel_ahead: bool, dt: f32) -> f32 {
        // Default: push gas
        let mut cmd = 1.0;

        // If strongly tilted backwards, brake to drop front
        if tilt > 20.0 {
            cmd = -1.0;
        }
        // If strongly downhill and nose diving, brake to lift nose
        else if slope < -10.0 && tilt < -10.0 {
            cmd = -0.6;
        }

        // If fuel is ahead, force gas
        if fuel_ahead {
            cmd = 1.0;
        }

        // Smooth
        self.throttle_smooth.step(cmd)
    }
}

// ============================== Main ==============================

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚗 Hill Climb Pro Bot — starting");

    // 1) Capture and layout
    let mut cap = FrameCapture::new()?;
    let (w, h) = cap.size();
    let vision = VisionConfig::for_frame(w, h);
    let touch = TouchPoints::for_frame(w, h);

    // 2) Assets
    let fuel_template = imgcodecs::imread("fuel_icon.jpg", imgcodecs::IMREAD_COLOR)?;
    if fuel_template.empty() {
        return Err(anyhow::anyhow!("fuel_icon.jpg not found or invalid"));
    }

    // 3) Modules
    let adb = AdbController::new(None);
    let mut policy = DecisionEngine::new();
    let mut fuel_ema = ExponentialSmooth::new(0.35);
    let mut airborne_det = AirborneDetector::new(200);

    // 4) Control loop
    let mut last = Instant::now();

    loop {
        let now = Instant::now();
        let dt = (now - last).as_secs_f32();
        last = now;

        let frame = cap.grab()?;

        // ROIs
        let fuel_roi = roi_clone(&frame, vision.fuel_roi)?;
        let car_roi = roi_clone(&frame, vision.car_roi)?;
        let gauge_roi = roi_clone(&frame, vision.gauge_roi)?;
        let ground = roi_clone(&frame, vision.ground_band)?;

        // Vision
        let fuel_ahead = detect_fuel_multiscale(&fuel_roi, &fuel_template)?;
        let tilt = estimate_car_tilt_deg(&car_roi)?;
        let slope = estimate_ground_slope_deg(&ground)?;
        let fuel_level = estimate_fuel_level_smooth(&gauge_roi, &mut fuel_ema)?;
        let airborne = airborne_det.detect(&car_roi)?;

        // Phase + throttle
        policy.update_phase(airborne, tilt);
        let mut throttle = policy.decide(tilt, slope, fuel_level, fuel_ahead, dt);

        // If fuel detected, bias positive to ensure pickup
        if fuel_ahead {
            throttle = throttle.max(0.5);
        }

        // Map throttle to press
        let duration_ms = (throttle.abs() * 220.0).clamp(60.0, 260.0) as i32;
        if throttle >= 0.10 {
            adb.long_press(touch.gas.0, touch.gas.1, duration_ms)
                .await?;
        } else if throttle <= -0.10 {
            adb.long_press(touch.brake.0, touch.brake.1, duration_ms)
                .await?;
        } else {
            // small dead zone: do nothing
        }

        println!(
            "phase={:?} tilt={:+5.1}° target={:+4.1}° slope={:+4.1}° fuel={:.2} fuel_ahead={} thr={:+.2} dur={}ms",
            policy.phase, tilt, policy.target_tilt, slope, fuel_level, fuel_ahead, throttle, duration_ms
        );

        // 12–15 Hz loop is enough (vision + control are light)
        sleep(Duration::from_millis(70)).await;
    }
}
