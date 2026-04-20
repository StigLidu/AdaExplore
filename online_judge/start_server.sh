#!/bin/bash
# Startup script for KernelBench Online Judge

set -e

# Default values
PORT=${PORT:-12017}
HOST=${HOST:-0.0.0.0}
WORKERS=${WORKERS:-1}
AVAILABLE_GPUS=${AVAILABLE_GPUS:-"0,1"}
GPU_ALLOCATION_MODE=${GPU_ALLOCATION_MODE:-"auto"}  # "auto" or "manual"
LOG_LEVEL=${LOG_LEVEL:-"debug"}  # uvicorn log level: debug, info, warning, error, critical

echo "Starting KernelBench Online Judge..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"
echo "Available GPUs: $AVAILABLE_GPUS"
echo "GPU Allocation Mode: $GPU_ALLOCATION_MODE"
echo "Log Level: $LOG_LEVEL"

# Try to get public IP address
echo ""
echo "=== Network Information ==="
# Try multiple methods to get public IP
PUBLIC_IP=""
if command -v curl &> /dev/null; then
    PUBLIC_IP=$(curl -s --max-time 2 https://api.ipify.org 2>/dev/null || echo "")
fi
if [ -z "$PUBLIC_IP" ] && command -v wget &> /dev/null; then
    PUBLIC_IP=$(wget -qO- --timeout=2 https://api.ipify.org 2>/dev/null || echo "")
fi

if [ -n "$PUBLIC_IP" ]; then
    echo "Public IP: $PUBLIC_IP"
    echo "Access URL: http://${PUBLIC_IP}:${PORT}"
    echo "Health check: http://${PUBLIC_IP}:${PORT}/health"
else
    echo "Could not determine public IP automatically"
    echo "Please check your server's public IP manually"
fi

# Get local IP addresses
echo ""
echo "Local IP addresses:"
if command -v hostname &> /dev/null; then
    hostname -I 2>/dev/null || ip addr show | grep "inet " | grep -v "127.0.0.1" | awk '{print $2}' | cut -d/ -f1 || echo "Could not determine local IP"
fi

echo ""
echo "Note: Make sure firewall allows incoming connections on port $PORT"
echo "      For example: sudo ufw allow $PORT/tcp"
echo ""

# Check if CUDA is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || echo "Warning: Could not check CUDA availability"

echo ""
echo "=== Starting Server ==="

# Run the server
AVAILABLE_GPUS="$AVAILABLE_GPUS" GPU_ALLOCATION_MODE="$GPU_ALLOCATION_MODE" exec python3 -m uvicorn online_judge.app_with_queue:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL"