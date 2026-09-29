"""
mayo_client.py
--------------
Drop-in LLM client for Mayo's Apigee -> Azure OpenAI gateway, with the SAME
interface as build_eval.VLLMClient (a `.choose(system, user) -> dict` method),
so build_eval / refine / skeleton_build / evaluator_probe work UNCHANGED — you
only swap which client you construct.

Handles the gateway's specifics:
  * OAuth2 client-credentials token, cached and auto-refreshed before expiry
  * model name lives in the URL path (/deployments/{engine}/...), not the body
  * gpt-5 family uses `max_completion_tokens` (not `max_tokens`)
  * JSON output via response_format={"type":"json_object"}
  * robust JSON parse via build_eval.balanced_json as a fallback

Setup:
  pip install requests python-dotenv
  .env must contain APIGEEX_CLIENT_ID and APIGEEX_SECRET_ID

Usage (mirrors VLLMClient):
  from mayo_client import MayoClient
  client = MayoClient(engine="gpt-5.5", env_path="/path/to/.env")
  client.choose(system_prompt, user_prompt)   # -> dict
"""
from __future__ import annotations
import os
import time
import requests

try:
    from build_eval import balanced_json
except Exception:                      # keep usable standalone
    import json as _json

    def balanced_json(text):
        s = text.find("{")
        if s < 0:
            return None
        depth = 0
        for i in range(s, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return _json.loads(text[s:i + 1])
                    except Exception:
                        return None
        return None


TOKEN_URL = "https://mcc.apix.mayo.edu/oauth/token"
APIGEE_BASE = "https://mcc.apix.mayo.edu"


class _TokenManager:
    """Fetches and caches the Apigee token; refreshes before it expires."""
    def __init__(self, client_id, secret_id, margin=60):
        self.client_id = client_id
        self.secret_id = secret_id
        self.margin = margin          # refresh this many secs before expiry
        self._token = None
        self._expires_at = 0.0

    def get(self) -> str:
        if self._token and time.time() < self._expires_at - self.margin:
            return self._token
        payload = (f"grant_type=client_credentials&client_id={self.client_id}"
                   f"&client_secret={self.secret_id}")
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        r = requests.post(TOKEN_URL, headers=headers, data=payload, timeout=30)
        r.raise_for_status()
        j = r.json()
        self._token = j["access_token"]
        # expires_in is seconds; default to 30min if absent
        self._expires_at = time.time() + float(j.get("expires_in", 1800))
        return self._token


class MayoClient:
    """Same interface as VLLMClient: .choose(system, user) -> dict."""
    def __init__(self, engine="gpt-5.5", env_path=None,
                 client_id=None, secret_id=None,
                 api_version="2024-10-21",
                 max_completion_tokens=16000,   # reasoning models spend most of
                 # the budget on hidden reasoning; 4000 left the
                 # actual JSON truncated on long prompts
                 temperature=1.0, top_p=1.0,
                 reasoning_effort=None,       # OFF by default; gateway may reject it
                 json_mode=False,             # OFF by default; some deployments 400
                 retries=3, timeout=90):
        if env_path:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        client_id = client_id or os.getenv("APIGEEX_CLIENT_ID")
        secret_id = secret_id or os.getenv("APIGEEX_SECRET_ID")
        if not client_id or not secret_id:
            raise RuntimeError("APIGEEX_CLIENT_ID / APIGEEX_SECRET_ID not set "
                               "(pass env_path or export them).")
        self.tokens = _TokenManager(client_id, secret_id)
        self.engine = engine
        self.url = (f"{APIGEE_BASE}/llm-azure-openai/openai/deployments/"
                    f"{engine}/chat/completions?api-version={api_version}")
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self.json_mode = json_mode
        self.retries = retries
        self.timeout = timeout

    def _body(self, system, user):
        b = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.json_mode:
            b["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            b["reasoning_effort"] = self.reasoning_effort
        return b

    def choose(self, system: str, user: str) -> dict:
        """Return a parsed JSON dict, or {'choice': None, '_error': ...}."""
        last = None
        for attempt in range(self.retries):
            try:
                headers = {"Authorization": f"Bearer {self.tokens.get()}"}
                r = requests.post(self.url, headers=headers,
                                  json=self._body(system, user),
                                  timeout=self.timeout)
                if r.status_code == 401:          # token rejected -> force refresh
                    self.tokens._token = None
                    raise RuntimeError("401; refreshing token")
                if r.status_code >= 400:
                    # surface the gateway's actual error message
                    raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
                r.raise_for_status()
                ch = r.json()["choices"][0]
                txt = ch["message"]["content"]
                fin = ch.get("finish_reason")
                obj = balanced_json(txt)
                if obj is not None:
                    return obj
                if fin == "length":
                    # reasoning models can burn the whole budget on hidden
                    # reasoning and emit nothing parseable
                    print(f"      [mayo] TRUNCATED (finish_reason=length, "
                          f"{len(txt)} chars returned). Raise "
                          f"max_completion_tokens (now "
                          f"{self.max_completion_tokens}).")
                last = f"unparseable (finish={fin}): {txt[:160]}"
            except Exception as e:                # noqa: BLE001
                last = str(e)[:200]
            time.sleep(1.5 * (attempt + 1))
        return {"choice": None, "_error": last}


if __name__ == "__main__":
    import sys
    env = sys.argv[1] if len(sys.argv) > 1 else None
    c = MayoClient(engine="gpt-5.5", env_path=env)
    print(c.choose('Respond ONLY JSON.',
                   'Return {"ok": true} and nothing else.'))