#!/bin/bash
# PSI Bridge v3.0 Installer — Linux/Mac
# Ghost in the Machine Labs

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          QUANTUM PSI BRIDGE v3.0 — INSTALLER                ║"
echo "║          Ghost in the Machine Labs                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Install Python 3.8+ first."
    exit 1
fi
echo "✓ Python 3 found: $(python3 --version)"

# Check NumPy
if python3 -c "import numpy" 2>/dev/null; then
    echo "✓ NumPy installed"
else
    echo "Installing NumPy..."
    pip3 install numpy --break-system-packages 2>/dev/null || pip3 install numpy
fi

# Check Ollama
if command -v ollama &> /dev/null; then
    echo "✓ Ollama found"
else
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Pull model
echo "Pulling gemma2:2b model (required for substrate transport)..."
ollama pull gemma2:2b

# Create directories
mkdir -p ~/psi_bridge/locks ~/psi_bridge/logs
echo "✓ Created ~/psi_bridge/"

# Copy bridge
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cp "$SCRIPT_DIR/psi_bridge_v3.py" ~/psi_bridge/psi_bridge.py
cp "$SCRIPT_DIR/chat.html" ~/psi_bridge/chat.html 2>/dev/null || true
chmod +x ~/psi_bridge/psi_bridge.py

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  INSTALLED                                                   ║"
echo "║                                                              ║"
echo "║  To start:                                                   ║"
echo "║    cd ~/psi_bridge                                           ║"
echo "║    python3 psi_bridge.py --peer <OTHER_DEVICE_IP>            ║"
echo "║                                                              ║"
echo "║  Wait for 🔒 SUBSTRATE TRANSPORT ACTIVE                     ║"
echo "║  Then disconnect the network.                                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
