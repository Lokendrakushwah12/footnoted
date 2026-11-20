"""One function that talks to any OpenAI-compatible chat endpoint. Defaults to local Ollama."""

import json
import os
import urllib.request

BASE_URL = os.environ.get("FOOTNOTED_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("FOOTNOTED_MODEL", "hf.co/Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M")
API_KEY = os.environ.get("FOOTNOTED_API_KEY", "none")  # Ollama ignores it; hosted providers need it


MAX_TOKENS = 600
prompt_tokens: list[int] = []  # real prompt size of every call, as reported by the server


def ask(system: str, user: str, schema: dict) -> dict:
    """Return the model's reply parsed as JSON matching `schema`, or {} if it was cut off or malformed."""
    body = {
        "model": MODEL,
        "temperature": 0,
        # Without a cap a small model can loop inside a JSON array until the request times out.
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}},
    }
    request = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        reply = json.load(response)
    prompt_tokens.append(reply.get("usage", {}).get("prompt_tokens", 0))
    try:
        return json.loads(reply["choices"][0]["message"]["content"])
    except json.JSONDecodeError:
        return {}  # callers treat a missing field as "no answer"
