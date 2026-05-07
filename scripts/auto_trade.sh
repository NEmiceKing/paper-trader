#!/bin/bash
# Auto-start paper trading when US market opens (Mon-Fri, 9:30 AM ET)
# Auto-stop when market closes (4:00 PM ET)
#
# US Eastern Time → Beijing Time: +12h (standard) / +13h (daylight)
# Market open:  9:30 AM ET = 21:30 Beijing (Mar-Nov) or 22:30 (Nov-Mar)
# Market close: 4:00 PM ET = 04:00 Beijing next day
#
# Cron entries (run: crontab -e):
#   Start: 25 21 * * 1-5  /path/to/scripts/auto_trade.sh start
#   Stop:   0  4  * * 2-6  /path/to/scripts/auto_trade.sh stop
#   Health: */30 * * * 1-5  /path/to/scripts/auto_trade.sh health

set -e

PROJECT_DIR="$HOME/paper-trader"
VENV_DIR="$PROJECT_DIR/.venv"
LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="$LOG_DIR/paper_trader.pid"
DEEPSEEK_KEY="${DEEPSEEK_API_KEY:-}"

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_DIR/auto_trade.log"
}

is_market_open() {
    # Check if US market is currently open (Mon-Fri, 9:30 AM - 4:00 PM ET)
    local day=$(date -u +%u)  # 1=Mon, 5=Fri, 6=Sat, 7=Sun
    local hour_utc=$(date -u +%H)
    local minute_utc=$(date -u +%M)

    # Weekend check
    if [ "$day" -gt 5 ]; then
        return 1
    fi

    # US market hours in UTC: 13:30 - 20:00 (ET+4/5, using UTC directly)
    # Actually: ET is UTC-4 (summer) or UTC-5 (winter). Open = 13:30/14:30 UTC, Close = 20:00/21:00 UTC
    # Simplification: check if hour is between 14-20 UTC (covers both summer and winter)
    local now_minutes=$((10#$hour_utc * 60 + 10#$minute_utc))

    # Market open window: 14:00-21:00 UTC (generous, covers DST changes)
    if [ "$now_minutes" -ge 840 ] && [ "$now_minutes" -le 1260 ]; then
        return 0
    fi
    return 1
}

is_running() {
    pgrep -f "src.main paper" > /dev/null 2>&1
}

start_trading() {
    if is_running; then
        log "Paper trading already running (PID: $(pgrep -f 'src.main paper' | tr '\n' ' '))"
    elif ! is_market_open; then
        log "Market closed — skipping paper trading start"
    else
        log "Starting paper trading..."
        source "$VENV_DIR/bin/activate"

        if [ -n "$DEEPSEEK_KEY" ]; then
            export DEEPSEEK_API_KEY="$DEEPSEEK_KEY"
        elif [ -f "$HOME/.deepseek_key" ]; then
            export DEEPSEEK_API_KEY=$(cat "$HOME/.deepseek_key")
        fi

        nohup python -m src.main paper >> "$LOG_DIR/trader.log" 2>&1 &
        local pid=$!
        echo "$pid" > "$PID_FILE"
        log "Paper trading started (PID: $pid, DEEPSEEK_KEY: ${DEEPSEEK_API_KEY:+set})"
    fi

    # Also start dashboard if not running
    if ! pgrep -f "streamlit run src/monitor/app.py" > /dev/null 2>&1; then
        log "Starting dashboard on port 8501..."
        nohup "$VENV_DIR/bin/streamlit" run src/monitor/app.py --server.port 8501 --server.headless true >> "$LOG_DIR/dashboard.log" 2>&1 &
        log "Dashboard started (PID: $!)"
    fi
}

stop_trading() {
    if is_running; then
        local pids=$(pgrep -f "src.main paper" | tr '\n' ' ')
        log "Stopping paper trading (PIDs: $pids)..."
        pkill -f "src.main paper" || true
        sleep 2
        rm -f "$PID_FILE"
        log "Paper trading stopped"
    else
        log "Paper trading not running"
    fi

    # Stop dashboard
    if pgrep -f "streamlit run src/monitor/app.py" > /dev/null 2>&1; then
        log "Stopping dashboard..."
        pkill -f "streamlit run src/monitor/app.py" || true
        log "Dashboard stopped"
    fi
}

health_check() {
    if is_market_open && ! is_running; then
        log "Health check FAILED: market open but paper trading not running. Auto-restarting..."
        start_trading
    elif is_market_open && is_running; then
        # Check if process is responsive (>10 min since last snapshot)
        local lp_file="$LOG_DIR/../logs/live_prices.json"
        if [ -f "$lp_file" ]; then
            local age=$(($(date +%s) - $(stat -f %m "$lp_file" 2>/dev/null || echo 0)))
            if [ "$age" -gt 900 ]; then
                log "Health check: live_prices stale (${age}s old), restarting..."
                stop_trading
                sleep 5
                start_trading
            fi
        fi
    fi
}

case "${1:-start}" in
    start)   start_trading ;;
    stop)    stop_trading ;;
    health)  health_check ;;
    status)
        if is_running; then
            echo "Paper trading RUNNING (PID: $(pgrep -f 'src.main paper'))"
            is_market_open && echo "Market: OPEN" || echo "Market: CLOSED"
        else
            echo "Paper trading STOPPED"
            is_market_open && echo "Market: OPEN — should be running!" || echo "Market: CLOSED"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|health|status}"
        exit 1
        ;;
esac
