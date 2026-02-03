#!/bin/bash

# Define paths
PROJECT_DIR="/Users/himarbravoperez/Desktop/Trabajo/Juego/GlobalMarketAnalyzer"
BRIDGE_PYTHON="/Users/himarbravoperez/Desktop/Trabajo/Juego/kernel_embeding_p/venv/bin/python"
ANALYZER_PYTHON="$PROJECT_DIR/.venv/bin/python"

# Kill previous instances (optional, use with care)
pkill -f "bridge_server.py"
pkill -f "telegram_commander.py"

cd "$PROJECT_DIR"

echo "=========================================="
echo "🚀 INICIANDO GLOBAL ECONOMIC ECOSYSTEM"
echo "=========================================="

# 1. Start AI Bridge (Background)
echo "🧠 Cargando AI Bridge (Qwen Model)..."
"$BRIDGE_PYTHON" bridge_server.py > logs/bridge.log 2>&1 &
BRIDGE_PID=$!
echo "   PID: $BRIDGE_PID"

# 2. Start Streamlit Dashboard (Background)
echo "📊 Iniciando Dashboard..."
"$ANALYZER_PYTHON" -m streamlit run main.py > logs/dashboard.log 2>&1 &
DASH_PID=$!

# 3. Start Telegram Commander (Foreground or Background? Let's keep it visible or background)
echo "🤖 Iniciando Telegram Commander..."
"$ANALYZER_PYTHON" telegram_commander.py

# Trap Ctrl+C to kill background processes
trap "kill $BRIDGE_PID $DASH_PID; exit" INT
