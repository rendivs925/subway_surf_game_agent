# Build stage
FROM rust:1.80-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    clang \
    libclang-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the entire workspace
COPY . .

# Build the backend
RUN cargo build --release -p gameagent-backend

# Runtime stage
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopencv-core4.5 \
    libopencv-imgcodecs4.5 \
    libopencv-imgproc4.5 \
    libopencv-objdetect4.5 \
    libopencv-highgui4.5 \
    && rm -rf /var/lib/apt/lists/*

# Copy the built binary
COPY --from=builder /app/target/release/gameagent-backend /usr/local/bin/

# Expose port
EXPOSE 9000

# Run the backend
CMD ["gameagent-backend"]