// subway_detect.rs (optimized for known objects in Subway Surfers)

use once_cell::sync::Lazy;
use opencv::{
    core::{self, Mat, Rect, Scalar, Size, Vector},
    imgcodecs::{imdecode, IMREAD_COLOR},
    imgproc::{
        self, canny, cvt_color, hough_lines_p, COLOR_BGR2GRAY, COLOR_BGR2HSV, MORPH_CLOSE,
        MORPH_ELLIPSE, MORPH_OPEN,
    },
    prelude::*,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::cmp::Ordering;
use std::sync::Mutex;
use std::time::{Duration, Instant};

// ========================= Config (Optimized for Subway Surfers) =========================

const LATENCY_BUDGET_MS: f64 = 100.0;
const GOLDEN_MIN_MS: f64 = 300.0;
const GOLDEN_MAX_MS: f64 = 850.0;
const TTC_HORIZON_MS: f64 = 2000.0;
const EMERGENCY_MS: f64 = 200.0;

// Overhead (barriers, signs)
const OVERHEAD_TOP_BAND_FRAC: f64 = 0.45;
const OVERHEAD_MIN_ASPECT: f64 = 1.5;
const OVERHEAD_MAX_REL_HEIGHT: f64 = 0.40;
const OVERHEAD_MIN_LEN_FRAC: f64 = 0.20;
const OVERHEAD_HOUGH_THRESH: i32 = 50;
const OVERHEAD_HOUGH_MAX_GAP: i32 = 25;
const OVERHEAD_CONFIRM_FRAMES: usize = 1;

// Ground (trains, barriers, stop blocks)
const GROUND_BAND_TOP_FRAC: f64 = 0.30;
const TRAIN_MIN_AREA_FRAC: f64 = 0.007;
const TRAIN_MIN_ASPECT: f64 = 1.2;
const TRAIN_MIN_BOTTOM_FRAC: f64 = 0.55;

const GROUND_BARRIER_MIN_AREA: f64 = 1500.0;
const GROUND_BARRIER_MAX_ASPECT: f64 = 1.2;
const SIDE_WALL_EDGE_FRAC: f64 = 0.18;
const SIDE_WALL_MIN_TALLNESS: f64 = 0.9;

// Collectibles (coins, keys, mystery boxes, power-ups)
const COLLECTIBLE_MIN_CONF: f64 = 0.65;
const COLLECTIBLE_CLUSTER_MIN: usize = 3;

// Tracking (improved for long runs)
const MIN_VELOCITY_PX_S: f64 = 30.0;
const TRACK_TTL_MS: u64 = 2500;

// NMS
const NMS_IOU_THRESH: f64 = 0.30;

// ========================= Types =========================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Detection {
    pub label: String,
    pub confidence: f64,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ActionPlan {
    Single {
        action: String,
        eta_ms: u64,
    },
    Sequence {
        actions: Vec<String>,
        eta_ms: Vec<u64>,
    },
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DetectionResponse {
    pub detections: Vec<Detection>,
    pub frame_width: i32,
    pub frame_height: i32,
    pub timestamp: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<ActionPlan>,
}

#[derive(Debug, Clone)]
pub enum DetectionMode {
    FaceDetection,
    SubwaySurfers,
    GeneralGame,
}

// ========================= Public API =========================

pub fn detect_objects(frame_bytes: &[u8]) -> String {
    detect_objects_with_mode(frame_bytes, DetectionMode::SubwaySurfers)
}

fn drift_toward_collectibles(resp: &DetectionResponse) -> Option<ActionPlan> {
    if resp.frame_width <= 0 || resp.frame_height <= 0 {
        return None;
    }

    let w = resp.frame_width as f64;
    let left_band_max = w * 0.45;
    let right_band_min = w * 0.55;

    let mut left_count = 0usize;
    let mut right_count = 0usize;

    for d in &resp.detections {
        let is_collectible = matches!(
            d.label.as_str(),
            "coin"
                | "magnet"
                | "jetpack"
                | "super_sneakers"
                | "hoverboard"
                | "key"
                | "mystery_box"
                | "gem"
                | "bonus"
        );
        if !is_collectible || d.confidence < COLLECTIBLE_MIN_CONF {
            continue;
        }

        let cx = d.x as f64 + (d.w as f64 / 2.0);
        if cx < left_band_max {
            left_count += 1;
        } else if cx > right_band_min {
            right_count += 1;
        }
    }

    if left_count >= COLLECTIBLE_CLUSTER_MIN && left_count > right_count + 1 {
        return Some(ActionPlan::Single {
            action: "swipe_left".into(),
            eta_ms: 0,
        });
    }
    if right_count >= COLLECTIBLE_CLUSTER_MIN && right_count > left_count + 1 {
        return Some(ActionPlan::Single {
            action: "swipe_right".into(),
            eta_ms: 0,
        });
    }

    None
}

pub fn detect_objects_for_game(frame_bytes: &[u8], game: &str) -> String {
    let result = match game {
        "subway" => detect_subway_surfers(frame_bytes),
        "hillclimb" => detect_hill_climb(frame_bytes),
        "general" => detect_general_game(frame_bytes),
        _ => detect_subway_surfers(frame_bytes),
    };

    match result {
        Ok(mut response) => {
            response.action = decide_action(game, &response);
            let action_str = response.action.as_ref().and_then(|ap| match ap {
                ActionPlan::Single { action, .. } => Some(action.clone()),
                ActionPlan::Sequence { actions, .. } => actions.get(0).cloned(),
            });
            let mut val = match serde_json::to_value(&response) {
                Ok(v) => v,
                Err(_) => {
                    json!({"detections":[],"frame_width":0,"frame_height":0,"timestamp": now_ms()})
                }
            };
            if let Some(a) = action_str {
                if let Some(map) = val.as_object_mut() {
                    map.insert("action".to_string(), serde_json::Value::String(a));
                }
            }
            serde_json::to_string(&val).unwrap_or_else(|_| create_empty_response())
        }
        Err(e) => {
            eprintln!("❌ Detection error for game '{}': {}", game, e);
            create_empty_response()
        }
    }
}

pub fn detect_objects_with_mode(frame_bytes: &[u8], mode: DetectionMode) -> String {
    let result = match mode {
        DetectionMode::FaceDetection => Ok(empty_resp(0, 0)),
        DetectionMode::SubwaySurfers => detect_subway_surfers(frame_bytes),
        DetectionMode::GeneralGame => detect_general_game(frame_bytes),
    };

    match result {
        Ok(mut response) => {
            response.action = decide_action("subway", &response);
            let action_str = response.action.as_ref().and_then(|ap| match ap {
                ActionPlan::Single { action, .. } => Some(action.clone()),
                ActionPlan::Sequence { actions, .. } => actions.get(0).cloned(),
            });
            let mut val = match serde_json::to_value(&response) {
                Ok(v) => v,
                Err(_) => {
                    json!({"detections":[],"frame_width":0,"frame_height":0,"timestamp": now_ms()})
                }
            };
            if let Some(a) = action_str {
                if let Some(map) = val.as_object_mut() {
                    map.insert("action".to_string(), serde_json::Value::String(a));
                }
            }
            serde_json::to_string(&val).unwrap_or_else(|_| create_empty_response())
        }
        Err(e) => {
            eprintln!("Detection error: {}", e);
            create_empty_response()
        }
    }
}

// ========================= Detection Core (Subway) =========================

fn detect_subway_surfers(
    frame_bytes: &[u8],
) -> Result<DetectionResponse, Box<dyn std::error::Error>> {
    let bgr = decode_bgr(frame_bytes)?;
    if bgr.empty() {
        return Ok(empty_resp(0, 0));
    }

    let (w, h) = (bgr.cols(), bgr.rows());

    let hsv = to_hsv(&bgr)?;
    let gray = to_gray(&bgr)?;

    let mut detections = Vec::<Detection>::new();
    detections.extend(detect_coins_and_key(&hsv, w, h)?);
    detections.extend(detect_powerups(&hsv, w, h)?);
    detections.extend(detect_mystery_boxes(&hsv, w, h)?);

    detections.extend(detect_ground_band(&gray, w, h)?);
    detections.extend(detect_overhead_hough(&gray, w, h)?);
    detections.extend(detect_mystery_hurdles(&gray, &hsv, w, h)?);

    detections = nms_and_normalize(detections, NMS_IOU_THRESH);

    Ok(DetectionResponse {
        detections,
        frame_width: w,
        frame_height: h,
        timestamp: now_ms(),
        action: None,
    })
}

// ========================= Detection Utilities =========================

fn decode_bgr(frame_bytes: &[u8]) -> opencv::Result<Mat> {
    let buf = Vector::from_slice(frame_bytes);
    imdecode(&buf, IMREAD_COLOR)
}

fn to_hsv(bgr: &Mat) -> opencv::Result<Mat> {
    let mut hsv = Mat::default();
    cvt_color(bgr, &mut hsv, COLOR_BGR2HSV, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    Ok(hsv)
}

fn to_gray(bgr: &Mat) -> opencv::Result<Mat> {
    let mut gray = Mat::default();
    cvt_color(bgr, &mut gray, COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    Ok(gray)
}

// ---------- Ground band detector (trains, ground barriers, side walls/sidewalk) ----------

fn detect_ground_band(gray: &Mat, w: i32, h: i32) -> opencv::Result<Vec<Detection>> {
    let top = (h as f64 * GROUND_BAND_TOP_FRAC) as i32;
    let band_h = (h - top).max(1);
    let roi = Rect::new(0, top, w.max(1), band_h);
    let gray_roi = Mat::roi(gray, roi)?;

    let mut edges = Mat::default();
    canny(&gray_roi, &mut edges, 70.0, 160.0, 3, false)?;
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(7, 7),
        core::Point::new(-1, -1),
    )?;
    let mut edges_d = Mat::default();
    imgproc::morphology_ex(
        &edges,
        &mut edges_d,
        MORPH_CLOSE,
        &kernel,
        core::Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;

    let comps = connected_components_stats(&edges_d)?;
    let mut cand = Vec::<Detection>::new();
    let frame_area = (w as f64) * (h as f64);

    for (x, y, ww, hh, area) in comps {
        let fy = y + top;
        let area_f = area as f64;

        if area_f < 1500.0 {
            continue;
        }

        let aspect = ww as f64 / (hh.max(1) as f64);
        let bottom = fy + hh;

        let near_left = x < (w as f64 * SIDE_WALL_EDGE_FRAC) as i32;
        let near_right = (x + ww) > (w as f64 * (1.0 - SIDE_WALL_EDGE_FRAC)) as i32;
        if (near_left || near_right)
            && (hh as f64) / (ww.max(1) as f64) > SIDE_WALL_MIN_TALLNESS
            && bottom > (h as f64 * 0.55) as i32
        {
            cand.push(Detection {
                label: "sidewalk".into(),
                confidence: 0.82,
                x,
                y: fy,
                w: ww,
                h: hh,
            });
            continue;
        }

        if area_f >= frame_area * TRAIN_MIN_AREA_FRAC
            && aspect >= TRAIN_MIN_ASPECT
            && bottom > (h as f64 * TRAIN_MIN_BOTTOM_FRAC) as i32
        {
            cand.push(Detection {
                label: "train".into(),
                confidence: 0.90,
                x,
                y: fy,
                w: ww,
                h: hh,
            });
            continue;
        }

        if area_f >= GROUND_BARRIER_MIN_AREA
            && aspect <= GROUND_BARRIER_MAX_ASPECT
            && bottom > (h as f64 * 0.55) as i32
        {
            cand.push(Detection {
                label: "barrier_ground".into(),
                confidence: 0.86,
                x,
                y: fy,
                w: ww,
                h: hh,
            });
            continue;
        }
    }

    Ok(nms_and_normalize(cand, NMS_IOU_THRESH))
}

// ---------- Overhead detector (top band + tolerant Hough) ----------

fn detect_overhead_hough(gray: &Mat, w: i32, h: i32) -> opencv::Result<Vec<Detection>> {
    let mut blurred = Mat::default();
    imgproc::gaussian_blur(
        gray,
        &mut blurred,
        Size::new(5, 5),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let mut edges = Mat::default();
    canny(&blurred, &mut edges, 60.0, 120.0, 3, false)?;

    let band_h = ((h as f64) * OVERHEAD_TOP_BAND_FRAC) as i32;
    if band_h <= 0 {
        return Ok(vec![]);
    }

    let roi = Rect::new(0, 0, w.max(1), band_h.max(1));
    let edges_roi = Mat::roi(&edges, roi)?;

    let min_len_primary = ((w as f64) * OVERHEAD_MIN_LEN_FRAC).max(30.0);

    let mut lines = Mat::default();
    hough_lines_p(
        &edges_roi,
        &mut lines,
        1.0,
        std::f64::consts::PI / 180.0,
        OVERHEAD_HOUGH_THRESH,
        min_len_primary,
        OVERHEAD_HOUGH_MAX_GAP as f64,
    )?;

    if lines.rows() == 0 {
        hough_lines_p(
            &edges_roi,
            &mut lines,
            1.0,
            std::f64::consts::PI / 180.0,
            (OVERHEAD_HOUGH_THRESH as f64 * 0.6) as i32,
            min_len_primary * 0.7,
            OVERHEAD_HOUGH_MAX_GAP as f64 * 1.6,
        )?;
    }

    let mut segments: Vec<(i32, i32, i32, i32)> = Vec::new();
    for r in 0..lines.rows() {
        let p = *lines.at_2d::<core::Vec4i>(r, 0)?;
        let (x1, y1, x2, y2) = (p[0], p[1], p[2], p[3]);
        let dx = (x2 - x1).abs();
        let dy = (y2 - y1).abs();
        if dx == 0 || dy * 4 > dx {
            continue;
        }
        if (dx as f64) < min_len_primary {
            continue;
        }
        segments.push((x1, y1, x2, y2));
    }
    if segments.is_empty() {
        return Ok(vec![]);
    }

    segments.sort_by_key(|s| s.1);
    let mut merged: Vec<(i32, i32, i32, i32)> = Vec::new();
    let mut cur = segments[0];
    let mut min_x = cur.0.min(cur.2);
    let mut max_x = cur.0.max(cur.2);
    let mut y_sum = cur.1 + cur.3;
    let mut count = 2;

    for &(x1, y1, x2, y2) in segments.iter().skip(1) {
        let y_mean = y_sum / count;
        if (y1 - y_mean).abs() <= 10 && (y2 - y_mean).abs() <= 10 {
            min_x = min_x.min(x1.min(x2));
            max_x = max_x.max(x1.max(x2));
            y_sum += y1 + y2;
            count += 2;
        } else {
            let y = y_sum / count;
            merged.push((min_x, y.saturating_sub(6), (max_x - min_x).max(1), 12));
            min_x = x1.min(x2);
            max_x = x1.max(x2);
            y_sum = y1 + y2;
            count = 2;
        }
    }
    let y = y_sum / count;
    merged.push((min_x, y.saturating_sub(6), (max_x - min_x).max(1), 12));

    let mut out = Vec::new();
    for (mx, my, mw, mh) in merged {
        let x = mx.clamp(0, w - 1);
        let y_full = my.clamp(0, band_h - 1);
        let aspect = (mw as f64) / (mh.max(1) as f64);
        let rel_h = (mh as f64) / (h as f64);
        if aspect < OVERHEAD_MIN_ASPECT || rel_h > OVERHEAD_MAX_REL_HEIGHT {
            continue;
        }

        out.push(Detection {
            label: "barrier_overhead".into(),
            confidence: 0.88,
            x,
            y: y_full,
            w: mw,
            h: mh,
        });
    }

    Ok(out)
}

fn detect_coins_and_key(hsv: &Mat, w: i32, h: i32) -> opencv::Result<Vec<Detection>> {
    let mut mask_yellow = Mat::default();
    core::in_range(
        hsv,
        &Scalar::new(20.0, 90.0, 90.0, 0.0),
        &Scalar::new(35.0, 255.0, 255.0, 0.0),
        &mut mask_yellow,
    )?;
    morph_open_close(&mut mask_yellow, 5)?;

    let components = connected_components_stats(&mask_yellow)?;
    let mut out = Vec::new();

    for (x, y, ww, hh, area) in components {
        let area_f = area as f64;
        if area_f < 100.0 || area_f > ((w * h) as f64 * 0.08) {
            continue;
        }
        let aspect = ww as f64 / (hh.max(1) as f64);
        if aspect >= 0.5 && aspect <= 1.8 {
            out.push(Detection {
                label: "coin".into(),
                confidence: 0.85,
                x,
                y,
                w: ww,
                h: hh,
            });
        }
        if area_f >= 300.0 && aspect > 1.4 {
            out.push(Detection {
                label: "key".into(),
                confidence: 0.70,
                x,
                y,
                w: ww,
                h: hh,
            });
        }
    }
    Ok(out)
}

fn detect_powerups(hsv: &Mat, w: i32, h: i32) -> opencv::Result<Vec<Detection>> {
    let mut out = Vec::new();
    out.extend(detect_color_blob(
        hsv,
        (0.0, 120.0, 120.0),
        (10.0, 255.0, 255.0),
        "magnet",
        0.75,
        w,
        h,
    )?);
    out.extend(detect_color_blob(
        hsv,
        (170.0, 120.0, 120.0),
        (180.0, 255.0, 255.0),
        "magnet",
        0.75,
        w,
        h,
    )?);
    out.extend(detect_color_blob(
        hsv,
        (100.0, 80.0, 80.0),
        (130.0, 255.0, 255.0),
        "jetpack",
        0.72,
        w,
        h,
    )?);
    out.extend(detect_color_blob(
        hsv,
        (45.0, 80.0, 80.0),
        (80.0, 255.0, 255.0),
        "super_sneakers",
        0.70,
        w,
        h,
    )?);
    out.extend(detect_color_blob(
        hsv,
        (130.0, 70.0, 70.0),
        (160.0, 255.0, 255.0),
        "hoverboard",
        0.68,
        w,
        h,
    )?);
    Ok(out)
}

fn detect_color_blob(
    hsv: &Mat,
    low: (f64, f64, f64),
    high: (f64, f64, f64),
    label: &str,
    conf: f64,
    w: i32,
    h: i32,
) -> opencv::Result<Vec<Detection>> {
    let mut mask = Mat::default();
    core::in_range(
        hsv,
        &Scalar::new(low.0, low.1, low.2, 0.0),
        &Scalar::new(high.0, high.1, high.2, 0.0),
        &mut mask,
    )?;
    morph_open_close(&mut mask, 5)?;
    let components = connected_components_stats(&mask)?;

    let mut out = Vec::new();
    for (x, y, ww, hh, area) in components {
        let area_f = area as f64;
        if area_f >= 120.0 && area_f < (w * h) as f64 * 0.05 {
            out.push(Detection {
                label: label.into(),
                confidence: conf,
                x,
                y,
                w: ww,
                h: hh,
            });
        }
    }
    Ok(out)
}

fn detect_mystery_boxes(hsv: &Mat, w: i32, h: i32) -> opencv::Result<Vec<Detection>> {
    let mut mask_blue = Mat::default();
    core::in_range(
        hsv,
        &Scalar::new(90.0, 80.0, 80.0, 0.0),
        &Scalar::new(140.0, 255.0, 255.0, 0.0),
        &mut mask_blue,
    )?;
    morph_open_close(&mut mask_blue, 4)?;

    let components = connected_components_stats(&mask_blue)?;
    let mut out = Vec::new();

    for (x, y, ww, hh, area) in components {
        let area_f = area as f64;
        if area_f > 200.0 && area_f < ((w * h) as f64 * 0.03) {
            let aspect = ww as f64 / hh as f64;
            if aspect > 0.8 && aspect < 1.2 {
                out.push(Detection {
                    label: "mystery_box".to_string(),
                    confidence: 0.80,
                    x,
                    y,
                    w: ww,
                    h: hh,
                });
            }
        }
    }
    Ok(out)
}

fn detect_mystery_hurdles(gray: &Mat, hsv: &Mat, w: i32, h: i32) -> opencv::Result<Vec<Detection>> {
    let top = (h as f64 * 0.4) as i32;
    let band_h = (h as f64 * 0.4) as i32;
    let roi = Rect::new(0, top, w, band_h);
    let gray_roi = Mat::roi(gray, roi)?;

    let mut edges = Mat::default();
    canny(&gray_roi, &mut edges, 60.0, 130.0, 3, false)?;

    let mut contours = Vector::<Vector<core::Point>>::new();
    imgproc::find_contours(
        &edges,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        core::Point::default(),
    )?;

    let mut detections = Vec::new();
    for contour in contours {
        let area = imgproc::contour_area(&contour, false)?;
        if area < 800.0 {
            continue;
        }
        let rect = imgproc::bounding_rect(&contour)?;
        let fy = rect.y + top;
        let aspect = rect.width as f64 / rect.height as f64;

        let label = if aspect > 2.0 {
            "mystery_overhead".to_string()
        } else if aspect < 0.5 {
            "mystery_tall".to_string()
        } else {
            "mystery_hurdle".to_string()
        };

        detections.push(Detection {
            label,
            confidence: 0.78,
            x: rect.x,
            y: fy,
            w: rect.width,
            h: rect.height,
        });
    }
    Ok(detections)
}

fn morph_open_close(mask: &mut Mat, k: i32) -> opencv::Result<()> {
    let kernel =
        imgproc::get_structuring_element(MORPH_ELLIPSE, Size::new(k, k), core::Point::new(-1, -1))?;
    let mut temp = Mat::default();
    imgproc::morphology_ex(
        mask,
        &mut temp,
        MORPH_OPEN,
        &kernel,
        core::Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    imgproc::morphology_ex(
        &temp,
        mask,
        MORPH_CLOSE,
        &kernel,
        core::Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    Ok(())
}

fn connected_components_stats(mask: &Mat) -> opencv::Result<Vec<(i32, i32, i32, i32, i32)>> {
    let mut mask_u8 = Mat::default();
    if mask.typ() != core::CV_8UC1 {
        mask.convert_to(&mut mask_u8, core::CV_8UC1, 1.0, 0.0)?;
    } else {
        mask_u8 = mask.clone();
    }

    let mut labels = Mat::default();
    let mut stats = Mat::default();
    let mut centroids = Mat::default();
    let count = imgproc::connected_components_with_stats(
        &mask_u8,
        &mut labels,
        &mut stats,
        &mut centroids,
        8,
        core::CV_32S,
    )?;

    let stats1: Mat = if stats.typ() != core::CV_32SC1 {
        let mut tmp = Mat::default();
        stats.convert_to(&mut tmp, core::CV_32SC1, 1.0, 0.0)?;
        tmp
    } else {
        stats
    };

    let mut out = Vec::new();
    for i in 1..count {
        let left = *stats1.at_2d::<i32>(i, imgproc::CC_STAT_LEFT)?;
        let top = *stats1.at_2d::<i32>(i, imgproc::CC_STAT_TOP)?;
        let width = *stats1.at_2d::<i32>(i, imgproc::CC_STAT_WIDTH)?;
        let height = *stats1.at_2d::<i32>(i, imgproc::CC_STAT_HEIGHT)?;
        let area = *stats1.at_2d::<i32>(i, imgproc::CC_STAT_AREA)?;
        out.push((left, top, width, height, area));
    }
    Ok(out)
}

fn nms_and_normalize(mut items: Vec<Detection>, iou_thresh: f64) -> Vec<Detection> {
    if items.is_empty() {
        return items;
    }
    items.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(Ordering::Equal)
    });

    let mut picked: Vec<Detection> = Vec::new();
    'outer: for d in items {
        for p in &picked {
            if iou_det(&d, p) >= iou_thresh && label_equiv(&d.label, &p.label) {
                continue 'outer;
            }
        }
        let label = if d.label == "side_wall" {
            "sidewalk".to_string()
        } else {
            d.label.clone()
        };
        picked.push(Detection { label, ..d });
    }
    picked
}

fn iou_det(a: &Detection, b: &Detection) -> f64 {
    let ax1 = a.x as f64;
    let ay1 = a.y as f64;
    let ax2 = (a.x + a.w) as f64;
    let ay2 = (a.y + a.h) as f64;
    let bx1 = b.x as f64;
    let by1 = b.y as f64;
    let bx2 = (b.x + b.w) as f64;
    let by2 = (b.y + b.h) as f64;

    let inter_w = (ax2.min(bx2) - ax1.max(bx1)).max(0.0);
    let inter_h = (ay2.min(by2) - ay1.max(by1)).max(0.0);
    let inter = inter_w * inter_h;
    let area_a = (a.w.max(0) as f64) * (a.h.max(0) as f64);
    let area_b = (b.w.max(0) as f64) * (b.h.max(0) as f64);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

fn label_equiv(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    let set = [a, b];
    set.contains(&"side_wall") && set.contains(&"sidewalk")
}

fn is_hazard(label: &str) -> bool {
    matches!(
        label,
        "train"
            | "barrier_ground"
            | "barrier_overhead"
            | "sidewalk"
            | "side_wall"
            | "mystery_hurdle"
            | "mystery_overhead"
            | "mystery_tall"
    )
}

// ========================= Tracking & Decision =========================

#[derive(Clone)]
struct Track {
    label: String,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    last_center_y: f64,
    vel_y: f64,
    last_seen: Instant,
    seen_count: usize,
}

static SUBWAY_TRACKS: Lazy<Mutex<Vec<Track>>> = Lazy::new(|| Mutex::new(Vec::new()));

fn decide_action(game: &str, resp: &DetectionResponse) -> Option<ActionPlan> {
    if resp.detections.is_empty() || resp.frame_width <= 0 || resp.frame_height <= 0 {
        return None;
    }
    match game {
        "subway" => decide_subway_action(resp),
        "hillclimb" => decide_hill_action(resp),
        _ => decide_general_action(resp),
    }
}

fn decide_subway_action(resp: &DetectionResponse) -> Option<ActionPlan> {
    let frame_w = resp.frame_width as f64;
    let frame_h = resp.frame_height as f64;

    let tracks = update_tracks(resp);
    if tracks.is_empty() {
        if let Some(plan) = drift_toward_collectibles(resp) {
            return Some(plan);
        }
        return None;
    }

    if let Some(plan) = emergency_reflex(&tracks, frame_w, frame_h) {
        return Some(plan);
    }

    if let Some(plan) = pattern_multi_obstacles(&tracks, frame_w, frame_h) {
        return Some(plan);
    }
    if let Some(plan) = pattern_mystery_hurdles(&tracks, frame_w, frame_h) {
        return Some(plan);
    }
    if let Some(plan) = pattern_double_trains(&tracks, frame_w, frame_h) {
        return Some(plan);
    }
    if let Some(plan) = pattern_barrier_chains(&tracks) {
        return Some(plan);
    }
    if let Some(plan) = pattern_train_exit_to_barrier(&tracks) {
        return Some(plan);
    }

    earliest_single_hazard(&tracks, frame_w, frame_h)
}

fn update_tracks(resp: &DetectionResponse) -> Vec<Track> {
    let now = Instant::now();
    let mut tracks = SUBWAY_TRACKS.lock().unwrap();
    tracks.retain(|t| now.duration_since(t.last_seen) <= Duration::from_millis(TRACK_TTL_MS));

    for d in &resp.detections {
        if !is_hazard(&d.label) {
            continue;
        }
        let mut best_idx = None;
        let mut best_overlap = 0.0;
        for (i, t) in tracks.iter().enumerate() {
            if !label_equiv(&t.label, &d.label) {
                continue;
            }
            let overlap = iou_det(
                d,
                &Detection {
                    label: t.label.clone(),
                    confidence: 1.0,
                    x: t.x,
                    y: t.y,
                    w: t.w,
                    h: t.h,
                },
            );
            if overlap > best_overlap {
                best_overlap = overlap;
                best_idx = Some(i);
            }
        }

        let center_y = d.y as f64 + (d.h as f64 / 2.0);
        if let Some(idx) = best_idx.filter(|_| best_overlap >= 0.3) {
            let t = &mut tracks[idx];
            let dt = now.duration_since(t.last_seen).as_secs_f64();
            let vy = if dt > 0.0 {
                (center_y - t.last_center_y) / dt
            } else {
                t.vel_y
            };
            t.vel_y = 0.6 * t.vel_y + 0.4 * vy;
            t.x = d.x;
            t.y = d.y;
            t.w = d.w;
            t.h = d.h;
            t.last_center_y = center_y;
            t.last_seen = now;
            t.seen_count = (t.seen_count + 1).min(8);
        } else {
            tracks.push(Track {
                label: d.label.clone(),
                x: d.x,
                y: d.y,
                w: d.w,
                h: d.h,
                last_center_y: center_y,
                vel_y: 0.0,
                last_seen: now,
                seen_count: 1,
            });
        }
    }

    for t in tracks.iter_mut() {
        if now.duration_since(t.last_seen) > Duration::from_millis(140) && t.seen_count > 0 {
            t.seen_count -= 1;
        }
    }
    tracks.clone()
}

fn emergency_reflex(tracks: &[Track], frame_w: f64, frame_h: f64) -> Option<ActionPlan> {
    let mut best: Option<(&Track, f64)> = None;
    for t in tracks {
        let ttc_ms = ttc_seconds(t, frame_h) * 1000.0;
        if ttc_ms > EMERGENCY_MS {
            continue;
        }
        if t.label == "barrier_overhead" && t.seen_count < OVERHEAD_CONFIRM_FRAMES {
            continue;
        }
        if best.map_or(true, |(_, cur)| ttc_ms < cur) {
            best = Some((t, ttc_ms));
        }
    }
    if let Some((t, _)) = best {
        let cx = t.x as f64 + (t.w as f64 / 2.0);
        let action = match t.label.as_str() {
            "barrier_overhead" | "mystery_overhead" => "swipe_down",
            "barrier_ground" | "mystery_hurdle" => "swipe_up",
            "train" | "sidewalk" | "side_wall" | "mystery_tall" => {
                if cx < frame_w * 0.5 {
                    "swipe_right"
                } else {
                    "swipe_left"
                }
            }
            _ => "swipe_up",
        };
        Some(ActionPlan::Single {
            action: action.into(),
            eta_ms: 0,
        })
    } else {
        None
    }
}

fn pattern_double_trains(tracks: &[Track], frame_w: f64, frame_h: f64) -> Option<ActionPlan> {
    let (left, center, right) = lanes_snapshot(tracks, frame_w, frame_h);
    let e_t_l = earliest_ttc(&left.trains);
    let e_t_c = earliest_ttc(&center.trains);
    let e_t_r = earliest_ttc(&right.trains);

    if e_t_l.is_some() && e_t_c.is_some() && e_t_r.is_none() {
        return schedule_single_json("swipe_right", e_t_l.unwrap().min(e_t_c.unwrap()));
    }
    if e_t_c.is_some() && e_t_r.is_some() && e_t_l.is_none() {
        return schedule_single_json("swipe_left", e_t_c.unwrap().min(e_t_r.unwrap()));
    }
    if e_t_l.is_some() && e_t_r.is_some() && e_t_c.is_none() {
        return schedule_single_json("swipe_up", e_t_l.unwrap().min(e_t_r.unwrap()));
    }
    if e_t_l.is_some() && e_t_c.is_some() && e_t_r.is_some() {
        return schedule_single_json(
            "swipe_up",
            e_t_l.unwrap().min(e_t_c.unwrap()).min(e_t_r.unwrap()),
        );
    }
    None
}

fn pattern_barrier_chains(tracks: &[Track]) -> Option<ActionPlan> {
    let (left, center, right) = lanes_snapshot(tracks, 1.0, 1.0);
    let chain_ok = |a: Option<f64>, b: Option<f64>| -> Option<(f64, f64)> {
        match (a, b) {
            (Some(t1), Some(t2)) if (t2 - t1) >= 0.20 && (t2 - t1) <= 0.60 => Some((t1, t2)),
            _ => None,
        }
    };
    if let Some((t1, t2)) = chain_ok(
        earliest_ttc(&left.barriers_ground),
        earliest_ttc(&left.barriers_overhead),
    ) {
        return schedule_seq_json("swipe_up", t1, "swipe_down", t2);
    }
    if let Some((t1, t2)) = chain_ok(
        earliest_ttc(&center.barriers_ground),
        earliest_ttc(&center.barriers_overhead),
    ) {
        return schedule_seq_json("swipe_up", t1, "swipe_down", t2);
    }
    if let Some((t1, t2)) = chain_ok(
        earliest_ttc(&right.barriers_ground),
        earliest_ttc(&right.barriers_overhead),
    ) {
        return schedule_seq_json("swipe_up", t1, "swipe_down", t2);
    }

    if let Some((t1, t2)) = chain_ok(
        earliest_ttc(&left.barriers_overhead),
        earliest_ttc(&left.barriers_ground),
    ) {
        return schedule_seq_json("swipe_down", t1, "swipe_up", t2);
    }
    if let Some((t1, t2)) = chain_ok(
        earliest_ttc(&center.barriers_overhead),
        earliest_ttc(&center.barriers_ground),
    ) {
        return schedule_seq_json("swipe_down", t1, "swipe_up", t2);
    }
    if let Some((t1, t2)) = chain_ok(
        earliest_ttc(&right.barriers_overhead),
        earliest_ttc(&right.barriers_ground),
    ) {
        return schedule_seq_json("swipe_down", t1, "swipe_up", t2);
    }
    None
}

fn pattern_train_exit_to_barrier(tracks: &[Track]) -> Option<ActionPlan> {
    let (left, center, right) = lanes_snapshot(tracks, 1.0, 1.0);
    let exit_ok = |t_train: Option<f64>, t_bar: Option<f64>| -> Option<(f64, f64)> {
        match (t_train, t_bar) {
            (Some(tt), Some(tb)) if (tb - tt) >= 0.20 && (tb - tt) <= 0.50 => Some((tt, tb)),
            _ => None,
        }
    };
    if let Some((tt, tb)) = exit_ok(
        earliest_ttc(&left.trains),
        earliest_ttc(&left.barriers_overhead),
    ) {
        return schedule_seq_json("swipe_up", tt, "swipe_down", tb);
    }
    if let Some((tt, tb)) = exit_ok(
        earliest_ttc(&left.trains),
        earliest_ttc(&left.barriers_ground),
    ) {
        return schedule_single_json("swipe_up", tb);
    }
    if let Some((tt, tb)) = exit_ok(
        earliest_ttc(&center.trains),
        earliest_ttc(&center.barriers_overhead),
    ) {
        return schedule_seq_json("swipe_up", tt, "swipe_down", tb);
    }
    if let Some((tt, tb)) = exit_ok(
        earliest_ttc(&center.trains),
        earliest_ttc(&center.barriers_ground),
    ) {
        return schedule_single_json("swipe_up", tb);
    }
    if let Some((tt, tb)) = exit_ok(
        earliest_ttc(&right.trains),
        earliest_ttc(&right.barriers_overhead),
    ) {
        return schedule_seq_json("swipe_up", tt, "swipe_down", tb);
    }
    if let Some((tt, tb)) = exit_ok(
        earliest_ttc(&right.trains),
        earliest_ttc(&right.barriers_ground),
    ) {
        return schedule_single_json("swipe_up", tb);
    }
    None
}

fn pattern_multi_obstacles(tracks: &[Track], frame_w: f64, frame_h: f64) -> Option<ActionPlan> {
    let (left, center, right) = lanes_snapshot(tracks, frame_w, frame_h);
    let mut center_ground_ttcs = center.barriers_ground;
    if center_ground_ttcs.len() >= 2 {
        center_ground_ttcs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        if (center_ground_ttcs[1] - center_ground_ttcs[0]) > 0.15
            && (center_ground_ttcs[1] - center_ground_ttcs[0]) < 0.55
        {
            return schedule_seq_json(
                "swipe_up",
                center_ground_ttcs[0],
                "swipe_up",
                center_ground_ttcs[1],
            );
        }
    }
    // Add more for other types/lanes
    None
}

fn pattern_mystery_hurdles(tracks: &[Track], frame_w: f64, frame_h: f64) -> Option<ActionPlan> {
    let mut earliest_ttc = f64::MAX;
    let mut action = "swipe_up";
    let mut cx = 0.0;
    for t in tracks {
        if t.label.contains("mystery") {
            let ttc = ttc_seconds(t, frame_h);
            if ttc < earliest_ttc {
                earliest_ttc = ttc;
                cx = t.x as f64 + (t.w as f64 / 2.0);
                action = if t.label == "mystery_overhead" {
                    "swipe_down"
                } else if t.label == "mystery_tall" {
                    if cx < frame_w * 0.5 {
                        "swipe_right"
                    } else {
                        "swipe_left"
                    }
                } else {
                    "swipe_up"
                };
            }
        }
    }
    if earliest_ttc != f64::MAX {
        schedule_single_json(action, earliest_ttc)
    } else {
        None
    }
}

fn earliest_single_hazard(tracks: &[Track], frame_w: f64, frame_h: f64) -> Option<ActionPlan> {
    let mut best: Option<(&Track, f64)> = None;
    for t in tracks {
        let ttc_s = ttc_seconds(t, frame_h);
        let eta_ms = (ttc_s * 1000.0) - LATENCY_BUDGET_MS;
        if eta_ms < GOLDEN_MIN_MS || eta_ms > GOLDEN_MAX_MS {
            continue;
        }
        if t.label == "barrier_overhead" && t.seen_count < OVERHEAD_CONFIRM_FRAMES {
            continue;
        }
        if best.map_or(true, |(_, b)| ttc_s < b) {
            best = Some((t, ttc_s));
        }
    }
    if let Some((t, ttc_s)) = best {
        let cx = t.x as f64 + (t.w as f64 / 2.0);
        let action = match t.label.as_str() {
            "barrier_overhead" | "mystery_overhead" => "swipe_down",
            "barrier_ground" | "mystery_hurdle" => "swipe_up",
            "train" | "sidewalk" | "side_wall" | "mystery_tall" => {
                if cx < frame_w * 0.5 {
                    "swipe_right"
                } else {
                    "swipe_left"
                }
            }
            _ => "swipe_up",
        };
        schedule_single_json(action, ttc_s)
    } else {
        None
    }
}

// ---- Lanes & TTC ----

#[derive(Default, Clone)]
struct LaneState {
    trains: Vec<f64>,
    barriers_ground: Vec<f64>,
    barriers_overhead: Vec<f64>,
    side_walls: Vec<f64>,
    mystery: Vec<f64>,
}

fn lanes_snapshot(
    tracks: &[Track],
    frame_w: f64,
    frame_h: f64,
) -> (LaneState, LaneState, LaneState) {
    let mut left = LaneState::default();
    let mut center = LaneState::default();
    let mut right = LaneState::default();

    for t in tracks {
        let ttc_ms = ttc_seconds(t, frame_h) * 1000.0;
        if ttc_ms > TTC_HORIZON_MS {
            continue;
        }

        let cx = t.x as f64 + (t.w as f64 / 2.0);
        let lane = if cx < frame_w / 3.0 * 1.05 {
            &mut left
        } else if cx > frame_w / 3.0 * 1.95 {
            &mut right
        } else {
            &mut center
        };

        let ttc_s = ttc_ms / 1000.0;
        match t.label.as_str() {
            "train" => lane.trains.push(ttc_s),
            "barrier_ground" => lane.barriers_ground.push(ttc_s),
            "barrier_overhead" => lane.barriers_overhead.push(ttc_s),
            "sidewalk" | "side_wall" => lane.side_walls.push(ttc_s),
            label if label.contains("mystery") => lane.mystery.push(ttc_s),
            _ => {}
        }
    }
    (left, center, right)
}

fn earliest_ttc(ttcs: &[f64]) -> Option<f64> {
    ttcs.iter()
        .cloned()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
}

fn ttc_seconds(track: &Track, frame_h: f64) -> f64 {
    let velocity = track.vel_y.max(MIN_VELOCITY_PX_S);
    let distance = (frame_h - track.last_center_y).max(1.0);
    distance / velocity
}

fn schedule_single_json(action: &str, ttc_s: f64) -> Option<ActionPlan> {
    let ttc_ms = ttc_s * 1000.0;
    let eta_ms = ttc_ms - LATENCY_BUDGET_MS;
    if ttc_ms < GOLDEN_MIN_MS || ttc_ms > GOLDEN_MAX_MS || eta_ms <= 0.0 {
        return None;
    }
    Some(ActionPlan::Single {
        action: action.into(),
        eta_ms: eta_ms.round() as u64,
    })
}

fn schedule_seq_json(a1: &str, t1_s: f64, a2: &str, t2_s: f64) -> Option<ActionPlan> {
    let t1_ms = t1_s * 1000.0;
    let t2_ms = t2_s * 1000.0;
    let e1 = t1_ms - LATENCY_BUDGET_MS;
    let e2 = t2_ms - LATENCY_BUDGET_MS;

    let in_window = |t: f64| t >= GOLDEN_MIN_MS && t <= GOLDEN_MAX_MS;
    if !(in_window(t1_ms) && in_window(t2_ms)) || e1 <= 0.0 || e2 <= 0.0 {
        return None;
    }

    Some(ActionPlan::Sequence {
        actions: vec![a1.into(), a2.into()],
        eta_ms: vec![e1.round() as u64, e2.round() as u64],
    })
}

// ========================= Other Game Stubs =========================

fn detect_hill_climb(frame_bytes: &[u8]) -> Result<DetectionResponse, Box<dyn std::error::Error>> {
    Ok(empty_resp(0, 0))
}

fn detect_general_game(
    frame_bytes: &[u8],
) -> Result<DetectionResponse, Box<dyn std::error::Error>> {
    Ok(empty_resp(0, 0))
}

fn decide_hill_action(resp: &DetectionResponse) -> Option<ActionPlan> {
    None
}

fn decide_general_action(resp: &DetectionResponse) -> Option<ActionPlan> {
    None
}

// ========================= Helpers =========================

fn empty_resp(w: i32, h: i32) -> DetectionResponse {
    DetectionResponse {
        detections: vec![],
        frame_width: w,
        frame_height: h,
        timestamp: now_ms(),
        action: None,
    }
}

fn create_empty_response() -> String {
    json!({ "detections": [], "frame_width": 0, "frame_height": 0, "timestamp": now_ms() })
        .to_string()
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
