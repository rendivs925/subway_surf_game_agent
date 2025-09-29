use std::time::Instant;

use actix::{Actor, StreamHandler};
use actix_web::{web, App, Error, HttpRequest, HttpResponse, HttpServer};
use actix_web_actors::ws;
use anyhow::Result;
use log::{debug, error, info, warn};
use once_cell::sync::Lazy;
use std::sync::{atomic::{AtomicBool, Ordering}, RwLock};

use crate::config::{with_config, update_latency_budget_ms, apply_profile};
use crate::detection::detect_objects_for_game;

struct WsSession {
    lat_samples: Vec<u64>,
}

impl Actor for WsSession {
    type Context = ws::WebsocketContext<Self>;
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WsSession {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        init_dump_cfg_once();
        match msg {
            Ok(ws::Message::Binary(bin)) => {
                match process_frame(&mut self.lat_samples, &bin) {
                    Ok(resp) => {
                        debug!("Sending response: {}", resp);
                        let _ = ctx.text(resp);
                    }
                    Err(e) => {
                        error!("process_frame error: {}", e);
                    }
                }
            }
            Ok(ws::Message::Text(text)) => {
                // Handle control messages, e.g., {"cmd":"set_dump","enabled":true,"dir":"dump"}
                match serde_json::from_str::<serde_json::Value>(&text) {
                    Ok(v) => {
                        if v.get("cmd").and_then(|c| c.as_str()) == Some("set_dump") {
                            if let Some(e) = v.get("enabled").and_then(|b| b.as_bool()) {
                                DUMP_ENABLED.store(e, Ordering::Relaxed);
                            }
                            if let Some(d) = v.get("dir").and_then(|d| d.as_str()) {
                                if let Ok(mut w) = DUMP_DIR.write() { *w = d.to_string(); }
                            }
                            let _ = ctx.text(r#"{"ok":true,"cmd":"set_dump"}"#);
                            info!("dump updated: enabled={} dir={}", DUMP_ENABLED.load(Ordering::Relaxed), DUMP_DIR.read().ok().map(|s| s.clone()).unwrap_or_default());
                        } else if v.get("command").and_then(|c| c.as_str()) == Some("set_game") {
                            if let Some(g) = v.get("game").and_then(|g| g.as_str()) {
                                apply_profile(g);
                                let msg = format!("{{\"ok\":true,\"cmd\":\"set_game\",\"game\":\"{}\"}}", g);
                                let _ = ctx.text(msg);
                                info!("applied profile for game='{}'", g);
                            }
                        }
                    }
                    Err(e) => warn!("bad text msg: {}", e),
                }
            }
            Ok(ws::Message::Ping(v)) => ctx.pong(&v),
            Ok(ws::Message::Close(r)) => {
                info!("ws close: {:?}", r);
                ctx.close(r)
            }
            Ok(_) => {}
            Err(e) => error!("ws error: {}", e),
        }
    }
}

fn process_frame(lat_samples: &mut Vec<u64>, jpeg: &[u8]) -> Result<String> {
    let t0 = Instant::now();

    // Update latest raw jpeg for /latest.jpg endpoint
    if let Ok(mut w) = LATEST_JPEG.write() {
        *w = jpeg.to_vec();
    }

    info!("📷 Processing frame ({} bytes)", jpeg.len());

    // Use the advanced detection system from detection.rs
    let detection_result = detect_objects_for_game(jpeg, "subway");

    // Parse the JSON response to extract action and detections
    let response: serde_json::Value = serde_json::from_str(&detection_result)
        .unwrap_or_else(|_| serde_json::json!({
            "action": "none",
            "detections": [],
            "frame_width": 0,
            "frame_height": 0
        }));

    let action = response.get("action")
        .and_then(|a| a.as_str())
        .unwrap_or("none");

    let detections = response.get("detections")
        .and_then(|d| d.as_array())
        .map(|arr| arr.len())
        .unwrap_or(0);

    let frame_width = response.get("frame_width")
        .and_then(|w| w.as_i64())
        .unwrap_or(0);

    let frame_height = response.get("frame_height")
        .and_then(|h| h.as_i64())
        .unwrap_or(0);

    // Latency telemetry (P95)
    let total_ms = t0.elapsed().as_millis() as u64;
    lat_samples.push(total_ms);
    if lat_samples.len() >= 20 {
        lat_samples.sort();
        let idx = ((lat_samples.len() as f64) * 0.95) as usize;
        let p95 = *lat_samples.get(idx.min(lat_samples.len() - 1)).unwrap_or(&total_ms);
        let old_budget = with_config(|c| c.latency_budget_ms);
        update_latency_budget_ms(p95.min(350));
        let new_budget = with_config(|c| c.latency_budget_ms);
        if new_budget != old_budget {
            info!("📊 Latency budget updated: {}ms → {}ms (p95: {}ms)",
                  old_budget, new_budget, p95);
        }
        lat_samples.clear();
    }

    // Enhanced logging
    info!("🎯 Action: {} | Detections: {} | Frame: {}×{} | Processing: {}ms",
          action, detections, frame_width, frame_height, total_ms);

    Ok(detection_result)
}

async fn ws_route(req: HttpRequest, stream: web::Payload) -> Result<HttpResponse, Error> {
    let peer = req.connection_info().peer_addr().unwrap_or("unknown").to_string();
    info!("📱 Android client connected from: {}", peer);
    let resp = ws::start(
        WsSession {
            lat_samples: Vec::with_capacity(64)
        },
        &req,
        stream,
    );
    info!("🚀 Advanced detection pipeline ready for {}", peer);
    resp
}

async fn index() -> HttpResponse {
    let body = r#"<!doctype html>
<html>
  <head>
    <meta charset='utf-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1, viewport-fit=cover'>
    <title>GameAgent Stream</title>
    <style>
      html, body { height:100%; margin:0; padding:0; background:#000; }
      #hud { position:fixed; top:8px; left:8px; z-index:2; color:#eee; font-family:sans-serif; opacity:0.8; pointer-events:none }
      #v { position:fixed; inset:0; width:100vw; height:100vh; object-fit:contain; background:#000; display:block }
      #fsbtn { position:fixed; bottom:12px; right:12px; z-index:3; background:#111; color:#eee; border:1px solid #444; border-radius:6px; padding:8px 12px; cursor:pointer; opacity:0.8 }
    </style>
  </head>
  <body>
    <div id='hud'>/latest.jpg (advanced detection system)</div>
    <img id='v' src='/latest.jpg' alt='preview'>
    <button id='fsbtn' title='Toggle Fullscreen (F)'>Fullscreen</button>
    <script>
      const img = document.getElementById('v');
      const btn = document.getElementById('fsbtn');
      function refresh(){ img.src = '/latest.jpg?ts=' + Date.now(); }
      setInterval(refresh, 120);
      function toggleFS(){
        if (!document.fullscreenElement) { document.documentElement.requestFullscreen().catch(()=>{}); }
        else { document.exitFullscreen().catch(()=>{}); }
      }
      btn.addEventListener('click', toggleFS);
      window.addEventListener('keydown', (e)=>{ if(e.key==='f' || e.key==='F'){ toggleFS(); } });
    </script>
  </body>
</html>"#;
    HttpResponse::Ok().content_type("text/html; charset=utf-8").body(body)
}

async fn latest_jpg() -> HttpResponse {
    if let Ok(r) = LATEST_JPEG.read() {
        if !r.is_empty() {
            return HttpResponse::Ok().content_type("image/jpeg").body(r.clone());
        }
    }
    HttpResponse::NoContent().finish()
}

async fn annotated_jpg() -> HttpResponse {
    // For now, return the same as latest_jpg since the advanced detection system
    // handles its own annotations
    latest_jpg().await
}

pub async fn run_server(addr: &str) -> std::io::Result<()> {
    info!("🚀 GameAgent Detection Backend v2.0 (Advanced)");
    info!("📡 WebSocket server listening on: {}", addr);
    info!("📱 Android endpoint: ws://{}/ws", addr);
    info!("🌐 Web interface: http://{}/", addr);
    info!("🎯 Detection pipeline: Advanced Subway Surfers + Pattern Recognition + Intelligent Planning");

    HttpServer::new(|| {
        App::new()
            .route("/ws", web::get().to(ws_route))
            .route("/", web::get().to(index))
            .route("/latest.jpg", web::get().to(latest_jpg))
            .route("/annotated.jpg", web::get().to(annotated_jpg))
            .route("/health", web::get().to(|| async { "OK" }))
    })
    .bind(addr)?
    .run()
    .await
}

// ===== Optional frame dump helpers =====
static DUMP_ENABLED: AtomicBool = AtomicBool::new(false);
static DUMP_DIR: Lazy<RwLock<String>> = Lazy::new(|| RwLock::new("dump".to_string()));

// Latest frame storage for /latest.jpg endpoint
static LATEST_JPEG: Lazy<RwLock<Vec<u8>>> = Lazy::new(|| RwLock::new(Vec::new()));

fn init_dump_cfg_once() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let on = std::env::var("BACKEND_DUMP").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false);
        let dir = std::env::var("BACKEND_DUMP_DIR").unwrap_or_else(|_| "dump".to_string());
        DUMP_ENABLED.store(on, Ordering::Relaxed);
        if let Ok(mut w) = DUMP_DIR.write() { *w = dir; }
    });
}