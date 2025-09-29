use once_cell::sync::Lazy;
use std::sync::RwLock;

#[derive(Clone, Debug)]
pub struct Config {
    // Timing
    pub golden_min_ms: u64,
    pub golden_max_ms: u64,
    pub emergency_ms: u64,
    pub latency_budget_ms: u64,

    // Vision
    pub crop_top_frac: f64,
    pub downscale_width: i32,
    pub canny_low: f64,
    pub canny_high: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            golden_min_ms: 400,
            golden_max_ms: 750,
            emergency_ms: 280,
            latency_budget_ms: 150,
            // Samsung Galaxy A15 5G (1080×2340) - crop status bar/notch area only
            crop_top_frac: 0.08, // Keep 92% of screen for gameplay detection
            // Higher resolution for better object detection on 1080p source
            downscale_width: 720,
            // Lower thresholds to pick up more edges for object detection
            canny_low: 20.0,
            canny_high: 60.0,
        }
    }
}

pub static CONFIG: Lazy<RwLock<Config>> = Lazy::new(|| RwLock::new(Config::default()));

pub fn with_config<T>(f: impl FnOnce(&Config) -> T) -> T {
    let cfg = CONFIG.read().expect("config poisoned");
    f(&cfg)
}

pub fn update_latency_budget_ms(new_ms: u64) {
    if let Ok(mut cfg) = CONFIG.write() {
        // Clamp to a reasonable window to avoid runaway.
        let clamped = new_ms.clamp(80, 350);
        cfg.latency_budget_ms = clamped;
    }
}

// Optional: apply preset profiles tuned for different games/modes
pub fn apply_profile(name: &str) {
    if let Ok(mut cfg) = CONFIG.write() {
        match name.to_lowercase().as_str() {
            // Subway Surfers on Galaxy A15 5G: keep most of the screen for gameplay
            "subway" | "subway_surfers" => {
                cfg.crop_top_frac = 0.10; // Minimal crop for UI/score area
                cfg.downscale_width = 720;
                cfg.canny_low = 20.0;
                cfg.canny_high = 60.0;
            }
            // Face detection placeholder: keep more of the upper region
            "face" | "face_detection" => {
                cfg.crop_top_frac = 0.10;
                cfg.downscale_width = 720;
                cfg.canny_low = 30.0;
                cfg.canny_high = 90.0;
            }
            // General: conservative crop
            _ => {
                cfg.crop_top_frac = 0.35;
                cfg.downscale_width = 800;
                cfg.canny_low = 40.0;
                cfg.canny_high = 100.0;
            }
        }
    }
}
