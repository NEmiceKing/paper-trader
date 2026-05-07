#!/bin/bash
# Paper Trader v2 — One-command setup for a fresh Mac
# Usage: bash scripts/setup.sh

set -e

echo "========================================"
echo " Paper Trader v2 — Setup"
echo "========================================"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ── Python environment ──
echo ""
echo "[1/5] Setting up Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -e ".[dev]" -q
pip install openai fastapi uvicorn streamlit -q
echo "  Done."

# ── Download initial data ──
echo ""
echo "[2/5] Downloading market data (37 symbols, ~30s)..."
python -m src.main download
echo "  Done."

# ── Train initial PPO model ──
echo ""
echo "[3/5] Training PPO model (200k timesteps, ~3 min)..."
python -m src.main train
echo "  Done."

# ── API key ──
echo ""
echo "[4/5] DeepSeek API Key"
if [ -n "$DEEPSEEK_API_KEY" ]; then
    echo "  Using DEEPSEEK_API_KEY from environment."
elif [ -f "$HOME/.deepseek_key" ]; then
    export DEEPSEEK_API_KEY=$(cat "$HOME/.deepseek_key")
    echo "  Using key from ~/.deepseek_key"
else
    echo "  Enter your DeepSeek API key (get one at https://platform.deepseek.com):"
    read -s DEEPSEEK_KEY
    echo "$DEEPSEEK_KEY" > "$HOME/.deepseek_key"
    chmod 600 "$HOME/.deepseek_key"
    export DEEPSEEK_API_KEY="$DEEPSEEK_KEY"
    echo "  Saved to ~/.deepseek_key"
fi

# ── Cron jobs ──
echo ""
echo "[5/5] Installing auto-trading cron jobs..."
chmod +x scripts/auto_trade.sh
(crontab -l 2>/dev/null | grep -v "auto_trade.sh" | grep -v "src.main refresh"; cat << 'CRON') | crontab -
# Paper Trader auto-trading
25 21 * * 1-5 $PROJECT_DIR/scripts/auto_trade.sh start
0 4 * * 2-6 $PROJECT_DIR/scripts/auto_trade.sh stop
*/30 14-20 * * 1-5 $PROJECT_DIR/scripts/auto_trade.sh health
30 16 * * 1-5 cd $PROJECT_DIR && $PROJECT_DIR/.venv/bin/python -m src.main refresh
CRON
echo "  Done."

echo ""
echo "========================================"
echo " Setup complete!"
echo "========================================"
echo ""
echo " What you need to do next:"
echo " 1. Install & start IB Gateway (https://www.interactivebrokers.com/en/trading/ibgateway.php)"
echo "    → Login with Paper Trading account"
echo "    → Settings → API → uncheck Read-Only API, port 4001"
echo ""
echo " 2. Start the Dashboard:"
echo "    cd $PROJECT_DIR && source .venv/bin/activate && make dashboard"
echo ""
echo " 3. Or let cron auto-start trading at 21:25 Beijing time"
echo ""
echo " Cron schedule:"
echo "   Start:  Mon-Fri 21:25 (9:25 AM ET)"
echo "   Stop:   Tue-Sat 04:00 (4:00 PM ET)"
echo "   Health: Every 30 min during market hours"
echo "   Refresh: Daily 16:30 (after close)"
