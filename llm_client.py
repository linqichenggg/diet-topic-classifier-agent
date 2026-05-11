import json
import os
import time
from functools import lru_cache
import urllib.error
import urllib.request


def _is_placeholder_value(value):
    text = str(value).strip()
    return not text or text.upper().startswith("YOUR_") or text.upper() in {"CHANGE_ME", "REPLACE_ME"}


def _load_local_secrets():
    path = os.path.join(os.path.dirname(__file__), "..", "secrets.local.json")
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def _resolve_config():
    secrets = _load_local_secrets()
    api_key = os.getenv("DEEPSEEK_API_KEY") or secrets.get("deepseek_api_key", "")
    if _is_placeholder_value(api_key):
        raise RuntimeError("Missing DeepSeek API key. Set DEEPSEEK_API_KEY or secrets.local.json.")

    base_url = os.getenv("DEEPSEEK_BASE_URL") or secrets.get(
        "deepseek_base_url", "https://api.deepseek.com/chat/completions"
    )
    model = os.getenv("DEEPSEEK_MODEL") or secrets.get("deepseek_model", "deepseek-chat")
    max_tokens = int(os.getenv("DEEPSEEK_MAX_TOKENS") or secrets.get("deepseek_max_tokens", 512))
    top_p = float(os.getenv("DEEPSEEK_TOP_P") or secrets.get("deepseek_top_p", 0.95))

    endpoint = str(base_url).rstrip("/")
    if "/chat/completions" not in endpoint:
        endpoint = f"{endpoint}/chat/completions"

    return {
        "api_key": str(api_key).strip(),
        "endpoint": endpoint,
        "model": str(model).strip(),
        "max_tokens": max(64, min(4096, max_tokens)),
        "top_p": max(0.1, min(1.0, top_p)),
    }


def chat_json(prompt, temperature=0.0, max_retries=3):
    config = _resolve_config()
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": "Return valid JSON only. Do not include Markdown."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "top_p": config["top_p"],
        "max_tokens": config["max_tokens"],
        "stream": False,
    }

    last_error = "Unknown error"
    for attempt in range(max_retries):
        try:
            request = urllib.request.Request(
                config["endpoint"],
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config['api_key']}",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                body = response.read().decode("utf-8")
            data = json.loads(body)
            content = data["choices"][0]["message"]["content"]
            return parse_json_content(content)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {e.code}: {detail}"
            if e.code in {400, 401, 403, 404}:
                break
        except Exception as e:
            last_error = str(e)
        if attempt < max_retries - 1:
            time.sleep(0.5)
    raise RuntimeError(f"Unable to get JSON response. {last_error}")


def parse_json_content(content):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        if start < 0:
            raise
        decoder = json.JSONDecoder()
        parsed, _ = decoder.raw_decode(content[start:])
        return parsed
