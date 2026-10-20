#!/usr/bin/env bash
# Install (or replace) the Hermes self-evolution LaunchAgent.
# Uses StartInterval (daily by default) instead of daily cron.
# Usage: ./scripts/setup_daily_cron.sh [interval_seconds]
#   Default interval: 86400 (24 hours)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INTERVAL="${1:-86400}"
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

mkdir -p "$REPO_DIR/logs" "$HOME/Library/LaunchAgents"
# Selection is deterministic and needs no Claude token. Do not embed credentials
# in a world-readable plist or change launchctl state from this setup script.
EXTRA_ENV="
        <key>PATH</key>
        <string>${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin</string>
        <key>HOME</key>
        <string>${HOME}</string>
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

# Parent operator loads the plist after review.

echo "LaunchAgent plist written (not loaded): ${PLIST_PATH}"
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
