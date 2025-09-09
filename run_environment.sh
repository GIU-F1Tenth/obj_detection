#!/bin/bash
# ROS2 LiDAR Object Detection - Environment Setup Script
# 
# This script sets up the complete environment needed to run the test suite

echo "🚀 Setting up ROS2 LiDAR Object Detection Environment..."

# Source ROS2 Humble
echo "📡 Sourcing ROS2 Humble..."
source /opt/ros/humble/setup.bash

# Activate Python virtual environment
echo "🐍 Activating Python virtual environment..."
source .venv/bin/activate

# Verify environment
echo "✅ Environment Status:"
echo "   ROS_DISTRO: $ROS_DISTRO"
echo "   Python: $(which python)"
echo "   Virtual Env: $VIRTUAL_ENV"

# Check key dependencies
echo "📦 Checking key dependencies..."
python -c "import numpy, scipy, sklearn, matplotlib; print('   ✅ All Python dependencies available')"

echo "🎯 Environment ready! You can now run:"
echo "   python run_tests.py                 # Run all tests"
echo "   python run_tests.py --verbose       # Detailed output"
echo "   python run_tests.py --performance   # Performance tests only"
echo ""