#!/bin/bash
# Approach 3.5 Demo Launcher
# Optimized Specialized Multi-Model - ~1.5s latency
# Usage: ./run_approach_3_5.sh [--mode gaming|real_world]

cd "$(dirname "$0")"
source venv/bin/activate
python demo/run_approach_3_5.py "$@"

