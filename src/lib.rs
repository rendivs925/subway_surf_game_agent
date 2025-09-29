pub mod config;
pub mod detection;

// Re-export main functions for external use
pub use detection::detect_objects_for_game;