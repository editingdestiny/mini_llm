#!/bin/bash
# Connect to traefik network on startup
docker network connect traefik_network mini-llm 2>/dev/null || true
exec docker-entrypoint.sh "$@"