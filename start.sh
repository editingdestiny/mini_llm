#!/bin/bash
# Start mini-llm container and connect to traefik network
docker run -d \
  --name mini-llm \
  --restart unless-stopped \
  -p 8503:8502 \
  -v /home/sd22750/mini-llm:/app \
  mini-llm:fixed3 \
  streamlit run /app/streamlit_app.py --server.port 8502 --server.address 0.0.0.0

# Connect to traefik network (traefik is on 172.18.0.0/16)
sleep 2
docker network connect traefik_network mini-llm 2>/dev/null || true
echo "Mini-LLM started and connected to traefik network"