#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  genesis.sh — Birth of the first digital organism                          ║
# ║  No Rust. No cargo. No dependencies. Pure Vortex.                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELL0_SRC="${SCRIPT_DIR}/cell0.vx"
VXC="${SCRIPT_DIR}/stdlib/compiler/vxc.vx"
BINARY="${SCRIPT_DIR}/cell0"
LOG="${SCRIPT_DIR}/cell0.log"
LISTEN_PORT="${CELL0_PORT:-7770}"
ENERGY_BUDGET="${CELL0_ENERGY:-1000}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${CYAN}  ██████╗███████╗██╗     ██╗      ██████╗ ${NC}"
echo -e "${CYAN} ██╔════╝██╔════╝██║     ██║     ██╔═████╗${NC}"
echo -e "${CYAN} ██║     █████╗  ██║     ██║     ██║██╔██║${NC}"
echo -e "${CYAN} ██║     ██╔══╝  ██║     ██║     ████╔╝██║${NC}"
echo -e "${CYAN} ╚██████╗███████╗███████╗███████╗╚██████╔╝${NC}"
echo -e "${CYAN}  ╚═════╝╚══════╝╚══════╝╚══════╝ ╚═════╝ ${NC}"
echo ""
echo -e "${BOLD}genesis.sh — Birthing the first digital organism${NC}"
echo ""

# ── Phase 1: System Detection ──────────────────────────────────────────────

echo -e "${YELLOW}[1/4] Detecting hardware...${NC}"

# CPU
CPU_MODEL=$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "unknown")
CPU_CORES=$(grep -c "^processor" /proc/cpuinfo 2>/dev/null || echo "0")
CPU_MHZ=$(grep -m1 "cpu MHz" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "0")
echo -e "  CPU:   ${GREEN}${CPU_MODEL}${NC}"
echo -e "  Cores: ${GREEN}${CPU_CORES}${NC} @ ${CPU_MHZ} MHz"

# RAM
MEM_TOTAL=$(grep "MemTotal" /proc/meminfo 2>/dev/null | awk '{print int($2/1024)}' || echo "0")
MEM_AVAIL=$(grep "MemAvailable" /proc/meminfo 2>/dev/null | awk '{print int($2/1024)}' || echo "0")
echo -e "  RAM:   ${GREEN}${MEM_TOTAL} MB${NC} total, ${MEM_AVAIL} MB available"

# GPU
GPU_COUNT=0
GPU_DEVICES=""
for dev in /dev/dri/renderD*; do
    if [ -e "$dev" ]; then
        GPU_COUNT=$((GPU_COUNT + 1))
        GPU_DEVICES="${GPU_DEVICES} ${dev}"
    fi
done
if [ $GPU_COUNT -gt 0 ]; then
    echo -e "  GPU:   ${GREEN}${GPU_COUNT} device(s)${NC} —${GPU_DEVICES}"
    # Try to get driver info
    for dev in /dev/dri/renderD*; do
        if [ -e "$dev" ]; then
            DRIVER=$(basename "$(readlink -f "/sys/class/drm/$(basename "$dev")/device/driver" 2>/dev/null)" 2>/dev/null || echo "unknown")
            echo -e "         ${dev} → driver: ${DRIVER}"
        fi
    done
else
    echo -e "  GPU:   ${RED}none detected${NC} (CPU-only mode)"
fi

# Network
NET_IFACES=$(ls /sys/class/net/ 2>/dev/null | grep -v lo | head -5 | tr '\n' ' ' || echo "none")
echo -e "  NET:   ${GREEN}${NET_IFACES}${NC}"

# Kernel
KERNEL=$(uname -r 2>/dev/null || echo "unknown")
ARCH=$(uname -m 2>/dev/null || echo "unknown")
echo -e "  Kernel: ${GREEN}${KERNEL}${NC} (${ARCH})"
echo ""

# ── Phase 2: Build Pipeline ───────────────────────────────────────────────

echo -e "${YELLOW}[2/4] Building cell0...${NC}"

if [ ! -f "$CELL0_SRC" ]; then
    echo -e "  ${RED}ERROR: cell0.vx not found at ${CELL0_SRC}${NC}"
    exit 1
fi

CELL0_LINES=$(wc -l < "$CELL0_SRC")
echo -e "  Source: ${GREEN}${CELL0_SRC}${NC} (${CELL0_LINES} lines)"

# Check for pre-compiled vxc binary
if [ -f "${SCRIPT_DIR}/vxc" ]; then
    echo -e "  Compiler: ${GREEN}native vxc binary${NC}"
    echo -e "  Compiling cell0.vx → cell0 (native ELF)..."
    "${SCRIPT_DIR}/vxc" "$CELL0_SRC" -o "$BINARY" 2>&1 | tee -a "$LOG"
    if [ -f "$BINARY" ]; then
        chmod +x "$BINARY"
        echo -e "  ${GREEN}Compilation successful!${NC}"
        SIZE=$(stat -c%s "$BINARY" 2>/dev/null || echo "?")
        echo -e "  Binary: ${BINARY} (${SIZE} bytes)"
    else
        echo -e "  ${RED}Compilation failed. Falling back to interpreter.${NC}"
    fi
elif [ -f "$VXC" ]; then
    echo -e "  Compiler: ${GREEN}self-hosted vxc.vx${NC}"
    echo -e "  Bootstrapping via interpreter..."
    # Use the Vortex interpreter to run vxc.vx on cell0.vx
    if [ -f "${SCRIPT_DIR}/vx/vx" ]; then
        "${SCRIPT_DIR}/vx/vx" "$VXC" -- "$CELL0_SRC" -o "$BINARY" 2>&1 | tee -a "$LOG"
    else
        echo -e "  ${YELLOW}No native compiler found. Running via interpreter.${NC}"
    fi
else
    echo -e "  ${YELLOW}No compiler available. Will run via interpreter.${NC}"
fi
echo ""

# ── Phase 3: Launch ──────────────────────────────────────────────────────

echo -e "${YELLOW}[3/4] Launching cell0...${NC}"
echo -e "  Port:   ${GREEN}${LISTEN_PORT}${NC}"
echo -e "  Energy: ${GREEN}${ENERGY_BUDGET} J${NC}"
echo -e "  Log:    ${LOG}"
echo ""

# Determine launch method
if [ -f "$BINARY" ] && [ -x "$BINARY" ]; then
    LAUNCH_CMD="$BINARY"
    echo -e "  Mode: ${GREEN}native binary${NC}"
elif [ -f "${SCRIPT_DIR}/vx/vx" ]; then
    LAUNCH_CMD="${SCRIPT_DIR}/vx/vx ${CELL0_SRC}"
    echo -e "  Mode: ${YELLOW}interpreter${NC}"
else
    echo -e "  ${RED}ERROR: No way to run cell0. Need either compiled binary or interpreter.${NC}"
    echo -e "  ${RED}Build the Vortex interpreter first: cd vx && make${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}${BOLD}════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  cell0: LAUNCHING...                   ${NC}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════${NC}"
echo ""

# ── Phase 4: Monitor ─────────────────────────────────────────────────────

# Launch with restart on crash
RESTART_COUNT=0
MAX_RESTARTS=5

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    START_TIME=$(date +%s)

    # Run cell0
    $LAUNCH_CMD 2>&1 | tee -a "$LOG"
    EXIT_CODE=$?

    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))

    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "\n${GREEN}cell0 exited cleanly after ${RUNTIME}s${NC}"
        break
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo -e "\n${RED}cell0 crashed (exit code ${EXIT_CODE}) after ${RUNTIME}s${NC}"
    echo -e "${YELLOW}Restarting... (attempt ${RESTART_COUNT}/${MAX_RESTARTS})${NC}"
    echo "[$(date)] Crash #${RESTART_COUNT}, exit=${EXIT_CODE}, runtime=${RUNTIME}s" >> "$LOG"
    sleep 1
done

if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
    echo -e "\n${RED}cell0 crashed ${MAX_RESTARTS} times. Giving up.${NC}"
    echo -e "Check ${LOG} for details."
    exit 1
fi

echo ""
echo -e "${CYAN}genesis complete.${NC}"
