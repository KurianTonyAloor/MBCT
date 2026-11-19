#!/bin/bash
# ========================================================
# MBCT Local API — cURL Command Examples
# ========================================================

# 1. OpenAI-Compatible Chat Completion (Non-Streaming)
echo "--- 1. Chat Completion ---"
curl -X POST http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-3-mini",
    "messages": [
      {"role": "user", "content": "Explain what a mutex is in one sentence."}
    ],
    "temperature": 0.5,
    "max_tokens": 50,
    "traits": {
      "directness": 0.9,
      "formality": 0.7
    }
  }'

echo -e "\n\n--- 2. Streaming Chat Completion (SSE) ---"
curl -N -X POST http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-3-mini",
    "messages": [
      {"role": "user", "content": "What are 3 benefits of indexing a database?"}
    ],
    "stream": true,
    "max_tokens": 60
  }'

echo -e "\n\n--- 3. Simple REST Generate Endpoint ---"
curl -X POST http://127.0.0.1:8088/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What database do we use?",
    "inject_memory": true
  }'

echo -e "\n\n--- 4. List Models ---"
curl -s http://127.0.0.1:8088/v1/models
