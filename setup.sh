#!/bin/bash

# W4A Environment Setup Script
# 
# Sets up the W4A reinforcement learning environment
# 
# Requirements:
# - Python 3.9 (required for SimulationInterface compatibility)
# - Virtual environment recommended
#
# Usage: ./setup.sh

set -e

echo "Setting up W4A Environment..."

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
if [[ "$PYTHON_VERSION" != "3.9" ]]; then
    echo "WARNING: Python 3.9 recommended for SimulationInterface compatibility"
    echo "Current version: $PYTHON_VERSION"
    echo "Continuing anyway..."
fi

echo "Installing SimulationInterface..."
# Cannot use -e mode due to compiled extensions
pip install ./SimulationInterface/
 
echo "Installing w4a dependencies..."
pip install -r requirements.txt

echo "Installing w4a in development mode..."
pip install -e .

echo "Running tests..."
python -m pytest tests/test_basic_multiagent.py -v

echo ""
echo "Setup complete"
echo ""
echo "Installed:"
echo "  - SimulationInterface (with compiled binaries)"
echo "  - w4a and dependencies"

