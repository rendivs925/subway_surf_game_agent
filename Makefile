# Subway Surfers Bot Makefile
# Production-ready automation for building, testing, and running the bot

.PHONY: help build run clean setup check-deps install-deps test bench format lint \
        check-device start-scrcpy stop-scrcpy android-setup model-check \
        release debug profile monitor logs backup restore docs

# Default target
.DEFAULT_GOAL := help

# Configuration
BINARY_NAME := subway-surfers-bot
MODEL_FILE := models/subway_surfers.onnx
BACKUP_DIR := backups
LOG_DIR := logs
DEVICE_ID := $(shell adb devices | grep -E "^\w+" | head -n1 | cut -f1)

# Colors for output
RED    := \033[0;31m
GREEN  := \033[0;32m
YELLOW := \033[0;33m
BLUE   := \033[0;34m
PURPLE := \033[0;35m
CYAN   := \033[0;36m
WHITE  := \033[0;37m
RESET  := \033[0m

# Help target
help: ## Show this help message
	@echo "$(CYAN)Subway Surfers Bot - Makefile Commands$(RESET)"
	@echo "$(YELLOW)========================================$(RESET)"
	@echo ""
	@echo "$(GREEN)Setup Commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /setup|install|deps/ {printf "  $(BLUE)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Build Commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /build|compile|release|debug/ {printf "  $(BLUE)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Run Commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /run|start|stop|monitor/ {printf "  $(BLUE)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Development Commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /test|format|lint|check|clean/ {printf "  $(BLUE)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Android Commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /android|device|scrcpy/ {printf "  $(BLUE)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Utility Commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && !/setup|install|deps|build|compile|release|debug|run|start|stop|monitor|test|format|lint|check|clean|android|device|scrcpy/ {printf "  $(BLUE)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Setup Commands
setup: check-deps install-deps create-dirs ## Complete initial setup
	@echo "$(GREEN)✅ Setup complete!$(RESET)"

check-deps: ## Check if all dependencies are installed
	@echo "$(CYAN)🔍 Checking dependencies...$(RESET)"
	@command -v cargo >/dev/null 2>&1 || { echo "$(RED)❌ Rust/Cargo not found$(RESET)"; exit 1; }
	@command -v adb >/dev/null 2>&1 || { echo "$(RED)❌ ADB not found$(RESET)"; exit 1; }
	@command -v scrcpy >/dev/null 2>&1 || { echo "$(RED)❌ scrcpy not found$(RESET)"; exit 1; }
	@pkg-config --exists opencv4 2>/dev/null || pkg-config --exists opencv 2>/dev/null || { echo "$(RED)❌ OpenCV not found$(RESET)"; exit 1; }
	@echo "$(GREEN)✅ All dependencies found$(RESET)"

install-deps: ## Install system dependencies (Ubuntu/Debian)
	@echo "$(CYAN)📦 Installing system dependencies...$(RESET)"
	@if command -v apt >/dev/null 2>&1; then \
		sudo apt update && sudo apt install -y libopencv-dev clang libclang-dev adb scrcpy; \
	elif command -v brew >/dev/null 2>&1; then \
		brew install opencv android-platform-tools scrcpy; \
	else \
		echo "$(YELLOW)⚠️  Please install dependencies manually for your system$(RESET)"; \
	fi

create-dirs: ## Create necessary directories
	@mkdir -p models $(BACKUP_DIR) $(LOG_DIR)
	@echo "$(GREEN)📁 Created directories: models, $(BACKUP_DIR), $(LOG_DIR)$(RESET)"

# Build Commands
build: check-deps ## Build the project in debug mode
	@echo "$(CYAN)🔨 Building $(BINARY_NAME) (debug)...$(RESET)"
	@cargo build
	@echo "$(GREEN)✅ Build complete$(RESET)"

release: check-deps ## Build optimized release version
	@echo "$(CYAN)🚀 Building $(BINARY_NAME) (release)...$(RESET)"
	@cargo build --release
	@echo "$(GREEN)✅ Release build complete$(RESET)"

debug: build ## Build with debug symbols and run
	@echo "$(CYAN)🐛 Starting debug build...$(RESET)"
	@cargo run

# Run Commands
run: release model-check check-device ## Run the bot (release mode)
	@echo "$(GREEN)🎮 Starting Subway Surfers Bot...$(RESET)"
	@echo "$(YELLOW)📱 Device: $(DEVICE_ID)$(RESET)"
	@cargo run --release 2>&1 | tee $(LOG_DIR)/bot-$(shell date +%Y%m%d-%H%M%S).log

start-bot: run ## Alias for run command

run-with-logs: release model-check check-device ## Run with detailed logging
	@echo "$(GREEN)🎮 Starting Subway Surfers Bot with detailed logging...$(RESET)"
	@RUST_LOG=debug cargo run --release 2>&1 | tee $(LOG_DIR)/bot-detailed-$(shell date +%Y%m%d-%H%M%S).log

# Android Commands
check-device: ## Check if Android device is connected
	@echo "$(CYAN)📱 Checking Android device connection...$(RESET)"
	@if [ -z "$(DEVICE_ID)" ]; then \
		echo "$(RED)❌ No Android device found. Please connect your device and enable USB debugging.$(RESET)"; \
		echo "$(YELLOW)💡 Run 'make android-setup' for help$(RESET)"; \
		exit 1; \
	else \
		echo "$(GREEN)✅ Device connected: $(DEVICE_ID)$(RESET)"; \
	fi

android-setup: ## Setup Android device for debugging
	@echo "$(CYAN)📱 Android Device Setup Instructions$(RESET)"
	@echo "$(YELLOW)=====================================$(RESET)"
	@echo "1. Enable Developer Options:"
	@echo "   - Go to Settings > About Phone"
	@echo "   - Tap 'Build Number' 7 times"
	@echo ""
	@echo "2. Enable USB Debugging:"
	@echo "   - Go to Settings > Developer Options"
	@echo "   - Enable 'USB Debugging'"
	@echo ""
	@echo "3. Connect via USB and authorize this computer"
	@echo ""
	@echo "4. Test connection:"
	@echo "   $(BLUE)make check-device$(RESET)"

start-scrcpy: check-device ## Start scrcpy in headless mode for frame capture
	@echo "$(CYAN)📺 Starting scrcpy for device $(DEVICE_ID)...$(RESET)"
	@echo "$(YELLOW)⚠️  This will start scrcpy in headless mode$(RESET)"
	@scrcpy -s $(DEVICE_ID) --no-display --record=- &
	@echo "$(GREEN)✅ scrcpy started$(RESET)"

stop-scrcpy: ## Stop all scrcpy processes
	@echo "$(CYAN)🛑 Stopping scrcpy processes...$(RESET)"
	@pkill -f scrcpy || true
	@echo "$(GREEN)✅ scrcpy processes stopped$(RESET)"

# Model Commands
model-check: ## Check if YOLO model exists
	@echo "$(CYAN)🧠 Checking YOLO model...$(RESET)"
	@if [ ! -f "$(MODEL_FILE)" ]; then \
		echo "$(RED)❌ YOLO model not found: $(MODEL_FILE)$(RESET)"; \
		echo "$(YELLOW)💡 Please place your trained YOLO model at $(MODEL_FILE)$(RESET)"; \
		echo "$(YELLOW)💡 Model should detect: player, coin, train_blocking, train_jumpable, train_free, barrier_overhead, barrier_ground$(RESET)"; \
		exit 1; \
	else \
		echo "$(GREEN)✅ Model found: $(MODEL_FILE)$(RESET)"; \
	fi

download-sample-model: ## Download a sample YOLO model (placeholder)
	@echo "$(CYAN)📥 Downloading sample model...$(RESET)"
	@echo "$(YELLOW)⚠️  This is a placeholder - you need to train your own model$(RESET)"
	@mkdir -p models
	@echo "$(RED)❌ No sample model available. Please train your own YOLO model.$(RESET)"

# Development Commands
test: ## Run all tests
	@echo "$(CYAN)🧪 Running tests...$(RESET)"
	@cargo test
	@echo "$(GREEN)✅ Tests complete$(RESET)"

bench: ## Run benchmarks
	@echo "$(CYAN)⚡ Running benchmarks...$(RESET)"
	@cargo bench
	@echo "$(GREEN)✅ Benchmarks complete$(RESET)"

format: ## Format code with rustfmt
	@echo "$(CYAN)🎨 Formatting code...$(RESET)"
	@cargo fmt
	@echo "$(GREEN)✅ Code formatted$(RESET)"

lint: ## Run clippy lints
	@echo "$(CYAN)🔍 Running lints...$(RESET)"
	@cargo clippy -- -D warnings
	@echo "$(GREEN)✅ Lints passed$(RESET)"

check: format lint test ## Run all code quality checks
	@echo "$(GREEN)✅ All checks passed$(RESET)"

clean: ## Clean build artifacts
	@echo "$(CYAN)🧹 Cleaning build artifacts...$(RESET)"
	@cargo clean
	@echo "$(GREEN)✅ Clean complete$(RESET)"

# Monitoring Commands
monitor: ## Monitor bot performance in real-time
	@echo "$(CYAN)📊 Monitoring bot performance...$(RESET)"
	@echo "$(YELLOW)Press Ctrl+C to stop$(RESET)"
	@watch -n 1 "ps aux | grep $(BINARY_NAME) | grep -v grep; echo ''; adb shell top -n 1 | head -20"

logs: ## Show recent bot logs
	@echo "$(CYAN)📋 Recent bot logs:$(RESET)"
	@ls -t $(LOG_DIR)/bot-*.log 2>/dev/null | head -5 | xargs -I {} sh -c 'echo "$(YELLOW)=== {} ===$(RESET)"; tail -20 {}'

tail-logs: ## Tail the most recent log file
	@echo "$(CYAN)📋 Tailing most recent log...$(RESET)"
	@tail -f $(shell ls -t $(LOG_DIR)/bot-*.log 2>/dev/null | head -1)

# Performance Commands
profile: ## Run with performance profiling
	@echo "$(CYAN)📈 Running with performance profiling...$(RESET)"
	@cargo build --release
	@perf record -g target/release/$(BINARY_NAME) || echo "$(YELLOW)⚠️  perf not available$(RESET)"

optimize: ## Build with maximum optimizations
	@echo "$(CYAN)⚡ Building with maximum optimizations...$(RESET)"
	@RUSTFLAGS="-C target-cpu=native" cargo build --release

# Backup Commands
backup: ## Backup configuration and logs
	@echo "$(CYAN)💾 Creating backup...$(RESET)"
	@mkdir -p $(BACKUP_DIR)
	@tar -czf $(BACKUP_DIR)/backup-$(shell date +%Y%m%d-%H%M%S).tar.gz \
		Cargo.toml src/ models/ $(LOG_DIR)/ Makefile README.md 2>/dev/null || true
	@echo "$(GREEN)✅ Backup created$(RESET)"

restore: ## List available backups
	@echo "$(CYAN)📋 Available backups:$(RESET)"
	@ls -la $(BACKUP_DIR)/*.tar.gz 2>/dev/null || echo "$(YELLOW)No backups found$(RESET)"

# Documentation Commands
docs: ## Generate and open documentation
	@echo "$(CYAN)📚 Generating documentation...$(RESET)"
	@cargo doc --open

# System Info
sysinfo: ## Show system information for debugging
	@echo "$(CYAN)💻 System Information$(RESET)"
	@echo "$(YELLOW)==================$(RESET)"
	@echo "OS: $(shell uname -s -r)"
	@echo "Rust: $(shell rustc --version 2>/dev/null || echo 'Not installed')"
	@echo "Cargo: $(shell cargo --version 2>/dev/null || echo 'Not installed')"
	@echo "ADB: $(shell adb version 2>/dev/null | head -1 || echo 'Not installed')"
	@echo "scrcpy: $(shell scrcpy --version 2>/dev/null || echo 'Not installed')"
	@echo "OpenCV: $(shell pkg-config --modversion opencv4 2>/dev/null || pkg-config --modversion opencv 2>/dev/null || echo 'Not found')"
	@echo ""
	@echo "$(CYAN)Android Devices:$(RESET)"
	@adb devices 2>/dev/null || echo "ADB not available"

# Quick start for new users
quickstart: ## Complete setup and run (new users)
	@echo "$(GREEN)🚀 Subway Surfers Bot - Quick Start$(RESET)"
	@echo "$(YELLOW)====================================$(RESET)"
	@make setup
	@make android-setup
	@echo ""
	@echo "$(GREEN)Next steps:$(RESET)"
	@echo "1. Place your YOLO model at: $(MODEL_FILE)"
	@echo "2. Connect your Android device via USB"
	@echo "3. Open Subway Surfers on your device"
	@echo "4. Run: $(BLUE)make run$(RESET)"

# Emergency stop
emergency-stop: ## Emergency stop all processes
	@echo "$(RED)🚨 Emergency stop - killing all related processes...$(RESET)"
	@pkill -f $(BINARY_NAME) || true
	@pkill -f scrcpy || true
	@adb kill-server || true
	@adb start-server || true
	@echo "$(GREEN)✅ Emergency stop complete$(RESET)"

# Install git hooks
install-hooks: ## Install git pre-commit hooks
	@echo "$(CYAN)🪝 Installing git hooks...$(RESET)"
	@echo '#!/bin/sh\nmake check' > .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "$(GREEN)✅ Git hooks installed$(RESET)"