#!/usr/bin/env bash
# free_memory.sh — stop co-tenant containers on the DGX Spark to reclaim memory
# before a heavy generation or training run.
#
# Usage: bash scripts/free_memory.sh
# Exit codes: 0 = OK (>=100 GiB available), 1 = below threshold
#
# Co-tenant policy (see docs/dgx-spark-regolith-manual.md):
#   STOP:  open-webui  ollama-compose  compose-arangodb-1
#   LEAVE: mjlab-dev   gsplat-dev      unsloth-dev   (ComfyUI systemd service)

set -euo pipefail

echo "=== regolith: free_memory.sh ==="
echo ""

echo "--- Stopping co-tenant containers ---"
# Use || true so we don't abort if a container is already stopped / doesn't exist.
docker stop open-webui ollama-compose compose-arangodb-1 || true

echo ""
echo "--- Memory after stop (free -h) ---"
free -h

# Numeric check: parse available memory in kibibytes from free -k
# free -k output (Linux):
#               total        used        free      shared  buff/cache   available
# Mem:       xxxxx        xxxxx        xxxxx        xxxxx        xxxxx        xxxxx
AVAIL_KB=$(free -k | awk '/^Mem:/ {print $7}')

if [ -z "${AVAIL_KB}" ]; then
    echo "" >&2
    echo "ERROR: Could not parse available memory from 'free -k'. Is this a Linux system?" >&2
    exit 1
fi

# Integer division: kibibytes -> gibibytes
AVAIL_GIB=$(( AVAIL_KB / 1024 / 1024 ))

echo ""
echo "Available memory: ${AVAIL_GIB} GiB"

THRESHOLD=100
if [ "${AVAIL_GIB}" -lt "${THRESHOLD}" ]; then
    echo "" >&2
    echo "ERROR: Available memory (${AVAIL_GIB} GiB) is below the ${THRESHOLD} GiB safety threshold." >&2
    echo "Check 'docker ps' for other co-tenant containers consuming memory." >&2
    echo "Restart stopped containers after your run: docker start open-webui ollama-compose compose-arangodb-1" >&2
    exit 1
fi

echo "OK — ${AVAIL_GIB} GiB available. Safe to proceed."
echo ""
echo "When your run is done, restart co-tenant containers:"
echo "  docker start open-webui ollama-compose compose-arangodb-1"
