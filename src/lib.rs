pub mod config;
pub mod detection;
pub mod ws;

// Re-export main functions for external use
pub use detection::detect_objects_for_game;
pub use ws::run_server;