"""
MBCT Local API Examples — Universal Integration Layer
Demonstrates how to call your steered local model from any Python application.
Supports:
  1. Standard OpenAI SDK (client = OpenAI(base_url="http://127.0.0.1:8088/v1", api_key="mbct"))
  2. Standard HTTP requests (urllib / requests)
  3. Dynamic per-request trait steering (dialing directness, formality, verbosity on the fly)
  4. Real-time token streaming
  5. External memory context injection
"""

import json
import urllib.request


BASE_URL = "http://127.0.0.1:8088"


def example_simple_generate():
    """Example 1: Call the simple REST API (/api/generate)."""
    print("\n--- Example 1: Simple Local API (/api/generate) ---")
    url = f"{BASE_URL}/api/generate"
    payload = {
        "prompt": "What database architecture should we use for our startup?",
        "traits": {
            "directness": 0.95,
            "formality": 0.85,
            "verbosity": 0.25
        },
        "max_tokens": 80,
        "inject_memory": True
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print(f"Model: {res.get('model')}")
        print(f"Steering Applied: {res.get('steering_applied')}")
        print(f"Memory Applied: {res.get('memory_applied')}")
        print(f"Response:\n{res.get('response')}\n")


def example_openai_chat_completions():
    """Example 2: Standard OpenAI-Compatible Chat Completions (/v1/chat/completions)."""
    print("\n--- Example 2: OpenAI-Compatible Chat Completions (/v1/chat/completions) ---")
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": "phi-3-mini",
        "messages": [
            {"role": "system", "content": "You are a concise technical advisor."},
            {"role": "user", "content": "Should we rewrite our backend in Go?"}
        ],
        "temperature": 0.3,
        "max_tokens": 100,
        "traits": {
            "directness": 1.0,
            "verbosity": 0.2
        }
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        msg = res["choices"][0]["message"]["content"]
        print(f"ID: {res.get('id')}")
        print(f"Assistant Content:\n{msg}")
        print(f"MBCT Metadata: {res.get('mbct')}\n")


def example_streaming_completions():
    """Example 3: Streaming Tokens via Server-Sent Events."""
    print("\n--- Example 3: Streaming Token by Token ---")
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": "phi-3-mini",
        "messages": [
            {"role": "user", "content": "List 3 rules for clean code."}
        ],
        "stream": True,
        "max_tokens": 60
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if not line_str or line_str == "data: [DONE]":
                continue
            if line_str.startswith("data: "):
                chunk_json = line_str[6:]
                chunk = json.loads(chunk_json)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
    print("\n")


def example_openai_python_sdk():
    """Example 4: Using the official 'openai' Python SDK (if installed)."""
    print("\n--- Example 4: Official OpenAI Python SDK ---")
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=f"{BASE_URL}/v1",
            api_key="mbct-local"  # Any dummy key is accepted
        )
        response = client.chat.completions.create(
            model="phi-3-mini",
            messages=[
                {"role": "user", "content": "Give me one piece of startup advice."}
            ],
            extra_body={
                "traits": {"directness": 0.9, "formality": 0.3}
            }
        )
        print("Response via openai library:")
        print(response.choices[0].message.content)
    except ImportError:
        print("openai package not installed (run 'pip install openai' to use the official SDK).")


if __name__ == "__main__":
    print("==================================================")
    print("   MBCT Local API Universal Client Test Suite    ")
    print("==================================================")
    print(f"Connecting to: {BASE_URL}")
    try:
        example_simple_generate()
        example_openai_chat_completions()
        example_streaming_completions()
        example_openai_python_sdk()
    except Exception as e:
        print(f"Error connecting to MBCT server: {e}")
        print("Make sure the MBCT server is running on http://127.0.0.1:8088")
