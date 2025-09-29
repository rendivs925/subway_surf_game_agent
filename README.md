# 🎮 Subway Surfers Bot

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/opencv-4.x-green.svg)](https://opencv.org)

A **100% local** automation bot for Subway Surfers that runs on your laptop and controls your Android phone via USB. Uses computer vision (YOLO) for object detection and ADB for touch controls.

## ✨ Features

- 🧠 **Local AI Processing**: All inference runs on your laptop - no cloud dependencies
- ⚡ **Real-time Detection**: 30-60 FPS frame capture and object detection
- 🎯 **Smart Decision Making**: Avoids obstacles, collects coins, chooses optimal paths
- 🔌 **USB Connection**: Uses ADB commands over USB - no wireless setup needed
- 🏗️ **Production Ready**: Robust error handling, performance monitoring, modular design
- 🛠️ **Make Integration**: Comprehensive Makefile for easy setup and development

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Complete setup for new users
make quickstart

# Place your YOLO model
cp /path/to/your/model.onnx models/subway_surfers.onnx

# Connect Android device and run
make run
```

### Option 2: Manual Setup

1. **Install dependencies**
   ```bash
   make install-deps  # Ubuntu/macOS automatic
   # OR manually install: Rust, OpenCV, ADB, scrcpy
   ```

2. **Setup Android device**
   ```bash
   make android-setup  # Shows setup instructions
   make check-device   # Verify connection
   ```

3. **Build and run**
   ```bash
   make release       # Build optimized version
   make run          # Run the bot
   ```

## 📋 Prerequisites

### Required Software
- 🦀 **Rust** (1.70+)
- 📷 **OpenCV** (4.x) development libraries
- 📱 **ADB** (Android Debug Bridge)
- 🖥️ **scrcpy** for screen mirroring
- 🧠 **YOLO Model** trained for Subway Surfers objects

### System Dependencies

Run `make install-deps` to automatically install on supported systems, or install manually:

<details>
<summary><strong>Ubuntu/Debian</strong></summary>

```bash
sudo apt update
sudo apt install libopencv-dev clang libclang-dev adb scrcpy
```
</details>

<details>
<summary><strong>macOS</strong></summary>

```bash
brew install opencv android-platform-tools scrcpy
```
</details>

<details>
<summary><strong>Windows</strong></summary>

Install manually:
- OpenCV via vcpkg or prebuilt binaries
- ADB from Android SDK Platform Tools
- scrcpy from GitHub releases
</details>

## 🛠️ Setup

### 1. Android Device Preparation

```bash
# Get setup instructions
make android-setup

# Quick checklist:
# ✓ Enable Developer Options (tap Build Number 7 times)
# ✓ Enable USB Debugging
# ✓ Connect via USB and authorize computer
# ✓ Verify with: make check-device
```

### 2. YOLO Model Setup

You need a YOLO model trained to detect these Subway Surfers objects:

| Class | Description |
|-------|-------------|
| `player` | The character |
| `coin` | Collectible coins |
| `train_blocking` | Trains that block the path (dodge) |
| `train_jumpable` | Low trains (jump over) |
| `train_free` | Safe areas on trains |
| `barrier_overhead` | High barriers (slide under) |
| `barrier_ground` | Ground barriers (jump over) |

```bash
# Verify model is in correct location
make model-check

# Expected location: models/subway_surfers.onnx
```

### 3. Build & Launch

```bash
# Check all dependencies
make check-deps

# Build release version
make release

# Run the bot (includes model check + device verification)
make run
```

## 🎮 Usage

### Starting the Bot

**Method 1: Full Automation (Recommended)**
```bash
make run  # Handles everything automatically
```

**Method 2: Manual Control**
```bash
# Terminal 1: Start screen capture
make start-scrcpy

# Terminal 2: Run bot
cargo run --release

# Stop when done
make stop-scrcpy
```

### Monitoring Performance

```bash
# Real-time monitoring
make monitor

# View logs
make logs

# Tail latest log
make tail-logs
```

### Expected Output

```
🎮 Subway Surfers Bot Starting...
📱 Device: ABC123DEF456
✅ All components initialized successfully
🚀 Starting game automation loop...
🎯 Executed action: MoveLeft
🎯 Executed action: Jump
📊 Stats: 100 frames, 29.8 FPS, 3.2 avg detections/frame
🎯 Executed action: MoveRight
```

## 🛠️ Available Make Commands

<details>
<summary><strong>Setup Commands</strong></summary>

| Command | Description |
|---------|-------------|
| `make setup` | Complete initial setup |
| `make install-deps` | Install system dependencies |
| `make check-deps` | Verify all dependencies |
| `make quickstart` | New user complete setup |

</details>

<details>
<summary><strong>Build Commands</strong></summary>

| Command | Description |
|---------|-------------|
| `make build` | Build debug version |
| `make release` | Build optimized release |
| `make clean` | Clean build artifacts |

</details>

<details>
<summary><strong>Run Commands</strong></summary>

| Command | Description |
|---------|-------------|
| `make run` | Run bot (release mode) |
| `make debug` | Run debug version |
| `make run-with-logs` | Run with detailed logging |

</details>

<details>
<summary><strong>Android Commands</strong></summary>

| Command | Description |
|---------|-------------|
| `make check-device` | Verify Android connection |
| `make android-setup` | Show setup instructions |
| `make start-scrcpy` | Start screen capture |
| `make stop-scrcpy` | Stop screen capture |

</details>

<details>
<summary><strong>Development Commands</strong></summary>

| Command | Description |
|---------|-------------|
| `make test` | Run all tests |
| `make format` | Format code |
| `make lint` | Run lints |
| `make check` | All quality checks |

</details>

<details>
<summary><strong>Monitoring Commands</strong></summary>

| Command | Description |
|---------|-------------|
| `make monitor` | Real-time performance |
| `make logs` | View recent logs |
| `make tail-logs` | Tail latest log |
| `make sysinfo` | System information |

</details>

<details>
<summary><strong>Emergency Commands</strong></summary>

| Command | Description |
|---------|-------------|
| `make emergency-stop` | Stop all processes |
| `make backup` | Backup configuration |

</details>

## 🏗️ Architecture

### Vision Module
- **FrameCapture**: Captures frames from scrcpy stream via OpenCV
- **YoloDetector**: Runs ONNX inference for object detection
- **Detection**: Parses YOLO outputs into structured data

### Control Module
- **AdbController**: Sends touch commands via ADB
- **Action**: Enum for game actions (Jump, Slide, MoveLeft, MoveRight)

### Decision Module
- **GameDecisionEngine**: Implements game logic and strategy
- **Lane tracking**: Monitors player position across 3 lanes
- **Threat avoidance**: Priority system for obstacle detection
- **Coin collection**: Opportunistic coin grabbing when safe

### Game Logic

1. **Threat Avoidance** (Highest Priority):
   - `train_blocking` → Move to adjacent lane
   - `train_jumpable` → Jump
   - `barrier_overhead` → Slide
   - `barrier_ground` → Jump

2. **Coin Collection** (Secondary Priority):
   - Move to lanes with coins if safe
   - Check for obstacles in target lane

3. **Action Cooldown**:
   - 300ms minimum between actions
   - Prevents spam and allows animations to complete

## Configuration

### Touch Coordinates (Adjustable in code)

```rust
// Jump: swipe up from bottom to top center
swipe(500, 1500, 500, 500, 150)

// Slide: swipe down from top to bottom center
swipe(500, 500, 500, 1500, 150)

// Move Left: swipe from right to left center
swipe(800, 1000, 200, 1000, 150)

// Move Right: swipe from left to right center
swipe(200, 1000, 800, 1000, 150)
```

### Detection Thresholds

```rust
// YOLO confidence threshold
let confidence_threshold = 0.5;

// Non-maximum suppression threshold
let nms_threshold = 0.4;

// Threat detection distance (40% of screen from bottom)
let threat_distance = screen_height * 0.4;

// Coin collection distance (60% of screen from bottom)
let coin_distance = screen_height * 0.6;
```

## 🔧 Troubleshooting

### Common Issues

<details>
<summary><strong>"Failed to initialize frame capture"</strong></summary>

**Solutions:**
```bash
# Check scrcpy status
make start-scrcpy

# Verify OpenCV installation
make check-deps

# Test with different capture method
scrcpy --no-display --record=test.mp4
```

**Common causes:**
- scrcpy not running or crashed
- OpenCV can't access video stream
- Camera/screen permissions denied
</details>

<details>
<summary><strong>"Failed to load YOLO model"</strong></summary>

**Solutions:**
```bash
# Verify model exists
make model-check

# Check file format
file models/subway_surfers.onnx

# Validate model structure
python -c "import onnx; onnx.checker.check_model('models/subway_surfers.onnx')"
```

**Common causes:**
- Model file missing or corrupted
- Wrong ONNX format/version
- Incorrect input/output dimensions
</details>

<details>
<summary><strong>"ADB command failed"</strong></summary>

**Solutions:**
```bash
# Check device connection
make check-device

# Reset ADB server
adb kill-server && adb start-server

# Verify USB debugging
make android-setup
```

**Common causes:**
- USB debugging disabled
- Device not authorized
- ADB server issues
</details>

<details>
<summary><strong>Poor Detection Accuracy</strong></summary>

**Solutions:**
- Retrain YOLO model with more diverse data
- Adjust confidence thresholds in code
- Ensure good lighting and screen quality
- Check model classes match game objects

**Tuning parameters:**
```rust
let confidence_threshold = 0.4; // Lower = more detections
let nms_threshold = 0.3;        // Lower = fewer duplicates
```
</details>

<details>
<summary><strong>Slow Performance</strong></summary>

**Diagnostics:**
```bash
# Monitor performance
make monitor

# Check system resources
make sysinfo

# Profile performance
make profile
```

**Optimizations:**
- Use smaller YOLO model (YOLOv5n/YOLOv8n)
- Close unnecessary applications
- Enable high-performance power mode
- Use SSD storage
- Optimize for GPU if available
</details>

### Emergency Recovery

```bash
# Stop everything if stuck
make emergency-stop

# Full reset
make clean && make setup && make run
```

## ⚡ Performance Optimization

### Model Optimization
```bash
# Use optimized build
make optimize  # Builds with target-cpu=native

# Profile performance
make profile   # Enables performance profiling
```

**Model selection tips:**
- **YOLOv5n/YOLOv8n**: Fastest, 5-15 FPS on CPU
- **YOLOv5s/YOLOv8s**: Balanced, 10-30 FPS on CPU
- **YOLOv5m/YOLOv8m**: Higher accuracy, 5-20 FPS on CPU
- **Quantized models**: 2-3x faster with minimal accuracy loss

### System Optimization

**Hardware recommendations:**
- 🖥️ **CPU**: 4+ cores, 3.0+ GHz
- 🧠 **RAM**: 8GB+ (16GB recommended)
- 💾 **Storage**: SSD for better I/O
- 🔌 **USB**: 3.0+ cable, avoid hubs

**Performance monitoring:**
```bash
# Real-time monitoring
make monitor

# Check FPS and resource usage
make run-with-logs | grep "Stats:"
```

### Connection Optimization
- Use USB 3.0+ cable (avoid USB 2.0)
- Connect directly to laptop (avoid USB hubs)
- Keep cable length under 2 meters
- Enable USB debugging performance mode

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup
```bash
# Fork the repo and clone
git clone https://github.com/yourusername/subway-surfers-bot
cd subway-surfers-bot

# Install git hooks
make install-hooks

# Run development checks
make check  # Format, lint, test
```

### Contribution Guidelines
1. 🍴 Fork the repository
2. 🌟 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. ✨ Make your changes
4. ✅ Run `make check` to ensure quality
5. 📝 Add tests if applicable
6. 💾 Commit your changes (`git commit -m 'Add amazing feature'`)
7. 📤 Push to branch (`git push origin feature/amazing-feature`)
8. 🔄 Open a Pull Request

### Development Commands
```bash
make format     # Format code
make lint       # Run lints
make test       # Run tests
make check      # All quality checks
make docs       # Generate documentation
```

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

**Free for personal and educational use.**

## ⚠️ Disclaimer

This bot is created for:
- 🎓 **Educational purposes** - Learning computer vision and automation
- 🔬 **Research purposes** - Studying game AI and decision making
- 🛠️ **Technical demonstration** - Showcasing local AI inference

**Important notes:**
- Use responsibly and ethically
- Respect game terms of service
- No warranty or guarantees provided
- Not affiliated with Subway Surfers or SYBO Games

## 🆘 Support

### Getting Help
- 📖 Check this README first
- 🔧 Try `make help` for command overview
- 🐛 Run `make sysinfo` for system diagnostics
- 📋 Check `make logs` for error details

### Reporting Issues
When reporting bugs, please include:
- Output of `make sysinfo`
- Relevant log files from `make logs`
- Steps to reproduce the issue
- Expected vs actual behavior

## 🙏 Acknowledgments

- **OpenCV** team for computer vision libraries
- **ONNX Runtime** for fast inference
- **Rust** community for excellent tooling
- **scrcpy** project for screen mirroring solution

---

**⭐ Star this repo if it helped you! ⭐**