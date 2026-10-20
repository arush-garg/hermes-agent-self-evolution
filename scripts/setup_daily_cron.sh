#!/usr/bin/env bash
# Install (or replace) the Hermes self-evolution LaunchAgent.
# Uses StartInterval (every 2 hours by default) instead of daily cron.
# Usage: ./scripts/setup_daily_cron.sh [interval_seconds]
#   Default interval: 7200 (2 hours)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INTERVAL="${1:-7200}"
PLIST_LABEL="com.hermes.skill-evolution"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_LABEL}.plist"

# Validate that INTERVAL is a positive integer
if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]] || [ "$INTERVAL" -le 0 ]; then
    echo "Error: interval must be a positive integer (seconds)" >&2
    exit 1
fi

if [ "$INTERVAL" -lt 3600 ]; then
    echo "Warning: interval of ${INTERVAL}s (${INTERVAL}s = less than 1 hour) may be aggressive."
    echo "  Recommended minimum: 7200 (2 hours)"
fi

# Prefer venv python, fall back to system python3
if [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
    PYTHON="$REPO_DIR/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="$(command -v python3)"
else
    echo "Error: no python3 found" >&2
    exit 1
fi

# --- Gather API keys from the user's shell environment ---
CLAUDE_TOKEN=""
if command -v zsh &>/dev/null; then
    CLAUDE_TOKEN="$(zsh -i -c 'echo "${CLAUDE_CODE_OAUTH_TOKEN:-}"' 2>/dev/null || true)"
fi

if [ -z "$CLAUDE_TOKEN" ]; then
    echo "WARNING: CLAUDE_CODE_OAUTH_TOKEN not found in zsh login shell." >&2
    echo "  Skill selection via Hermes CLI will fall back to lowest-scored metric." >&2
    echo "  Set 'export CLAUDE_CODE_OAUTH_TOKEN=...' in your ~/.zshrc and re-run this script." >&2
else
    echo "  Found CLAUDE_CODE_OAUTH_TOKEN (${#CLAUDE_TOKEN} chars)"
fi

mkdir -p "$REPO_DIR/logs"

# Unload existing agent if present
if launchctl list "$PLIST_LABEL" &>/dev/null; then
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
fi

# Build the EnvironmentVariables dict
EXTRA_ENV="
        <key>PATH</key>
        <string>${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin</string>
        <key>HOME</key>
        <string>${HOME}</string>"

if [ -n "$CLAUDE_TOKEN" ]; then
    EXTRA_ENV="$EXTRA_ENV
        <key>CLAUDE_CODE_OAUTH_TOKEN</key>
        <string>${CLAUDE_TOKEN}</string>"
fi

# Omniroute: route all DSPy/LiteLLM calls through the local proxy
EXTRA_ENV="$EXTRA_ENV
        <key>HERMES_LLM_PROVIDER</key>
        <string>omniroute</string>"

# Write the plist
cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON}</string>
        <string>-m</string>
        <string>evolution.monitor.daily_runner</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${REPO_DIR}</string>
    <key>StartInterval</key>
    <integer>${INTERVAL}</integer>
    <key>RunAtLoad</key>
    <true/>
    <key>EnvironmentVariables</key>
    <dict>${EXTRA_ENV}
    </dict>
    <key>StandardOutPath</key>
    <string>${REPO_DIR}/logs/daily_evolution_launchd.log</string>
    <key>StandardErrorPath</key>
    <string>${REPO_DIR}/logs/daily_evolution_launchd_err.log</string>
</dict>
</plist>
EOF

launchctl load "$PLIST_PATH"

# Remove any legacy crontab entry from the old setup
EXISTING_CRON="$(crontab -l 2>/dev/null || true)"
if echo "$EXISTING_CRON" | grep -q "hermes-skill-evolution-daily"; then
    echo "$EXISTING_CRON" | grep -v "hermes-skill-evolution-daily" | crontab -
    echo "Removed legacy crontab entry."
fi

# Also delete the old daily-only run flag so the gauntlet starts fresh
STATE_FILE="${REPO_DIR}/logs/evolution_schedule.json"
if [ -f "$STATE_FILE" ] && [ ! -s "$STATE_FILE" ]; then
    rm -f "$STATE_FILE"
fi

echo "LaunchAgent installed: ${PLIST_PATH}"
echo "  Python:  ${PYTHON}"
echo "  Module:  evolution.monitor.daily_runner"
echo "  Interval: every ${INTERVAL}s ($((INTERVAL / 3600))h $(( (INTERVAL % 3600) / 60 ))m)"
echo "  RunAtLoad: true (fires immediately on login)"
echo "  Logs:    ${REPO_DIR}/logs/"
echo ""
echo "Next steps:"
echo "  Verify:   launchctl list ${PLIST_LABEL}"
echo "  Run now:  ${PYTHON} -m evolution.monitor.daily_runner"
echo "  Dry-run:  ${PYTHON} -m evolution.monitor.daily_runner --dry-run"
echo "  Force:    ${PYTHON} -m evolution.monitor.daily_runner --force-pipeline skills --force-artifact linear"
echo "  Watch:    tail -f ${REPO_DIR}/logs/daily_runner_latest.log"
echo "  Logs:     ls -lt ${REPO_DIR}/logs/"
