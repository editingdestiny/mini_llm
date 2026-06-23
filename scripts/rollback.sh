#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Rollback mini-llm to previous image tag
# Usage: ./rollback.sh [sha | previous]
# ──────────────────────────────────────────────────────────────
set -euo pipefail

REGISTRY="ghcr.io/editingdestiny/mini-llm"
COMPOSE_FILE="/home/sd22750/mini-llm/docker-compose.yml"
HEALTH_URL="https://mini-llm.sd-ai.co.uk"

# Resolve target image
if [[ "${1:-}" == "previous" ]] || [[ -z "${1:-}" ]]; then
    # Fetch the second-latest SHA tag from GHCR
    TARGET=$(curl -s -f -H "Authorization: Bearer $(cat ~/.github_token 2>/dev/null || echo '')" \
        "https://ghcr.io/v2/editingdestiny/mini-llm/tags/list" 2>/dev/null \
        | python3 -c "import sys,json; tags=sorted(json.load(sys.stdin)); print(tags[-2] if len(tags)>1 else tags[-1])" 2>/dev/null \
        || echo "previous")
else
    TARGET="$1"
fi

if [[ "$TARGET" == "previous" ]] || [[ -z "$TARGET" ]]; then
    echo "ERROR: Could not resolve target image tag"
    exit 1
fi

IMAGE="${REGISTRY}:${TARGET}"
echo "🔄 Rolling back to: $IMAGE"

# Pull image
docker pull "$IMAGE"

# Update compose to use the target image
COMPOSE_BACKUP="${COMPOSE_FILE}.bak"
cp "$COMPOSE_FILE" "$COMPOSE_BACKUP"
sed -i "s|image: ${REGISTRY}:.*|image: ${IMAGE}|" "$COMPOSE_FILE"

# Restart
docker-compose -f "$COMPOSE_FILE" up -d
sleep 20

# Health check
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" --max-time 20 "$HEALTH_URL" 2>/dev/null || echo "000")
if [[ "$HTTP_CODE" == "200" ]]; then
    echo "✅ Rollback successful — health check HTTP $HTTP_CODE"
else
    echo "❌ Health check failed (HTTP $HTTP_CODE) — restoring compose"
    cp "$COMPOSE_BACKUP" "$COMPOSE_FILE"
    docker-compose -f "$COMPOSE_FILE" up -d
    exit 1
fi
