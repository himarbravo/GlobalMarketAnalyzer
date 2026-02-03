#!/bin/bash

# Global Market Analyst - Daily Runner
# Este script está diseñado para ser ejecutado por launchd (cron) automáticamente.

# 1. Definir rutas absolutas (CRICIAL para cron/launchd)
PROJECT_DIR="/Users/himarbravoperez/Desktop/Trabajo/Juego/GlobalMarketAnalyzer"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
SCRIPT_PATH="$PROJECT_DIR/market_analyst.py"
LOG_FILE="$PROJECT_DIR/logs/analyst.log"

# 2. Ir al directorio
cd "$PROJECT_DIR" || exit 1

# 3. Timestamp en el log
echo "------------------------------------------------" >> "$LOG_FILE"
echo "⏰ Starting Daily Analysis: $(date)" >> "$LOG_FILE"

# 4. Ejecutar el analista
"$VENV_PYTHON" "$SCRIPT_PATH" >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Success. Analysis complete." >> "$LOG_FILE"
    # Opcional: Enviar notificación nativa
    osascript -e 'display notification "Tu analista ha completado el informe diario." with title "Global Market Analyst"'
else
    echo "❌ Failed with exit code $EXIT_CODE" >> "$LOG_FILE"
    osascript -e 'display notification "Error al ejecutar el analista." with title "Global Market Analyst Error"'
fi
