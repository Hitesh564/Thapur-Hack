"""OpenAI vision-language client. Used only in preprocessing, never in the simulation loop.

Every response is JSON, validated against a Pydantic model. The key is read from the
OPENAI_API_KEY environment variable (or, on Windows, the user environment), never from files.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import threading
from pathlib import Path
from typing import Optional, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import get_config

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
T = TypeVar("T", bound=BaseModel)
_client = None
_lock = threading.Lock()


class VLMUnavailable(RuntimeError):
    pass


def _api_key() -> Optional[str]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key and sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                key = winreg.QueryValueEx(k, "OPENAI_API_KEY")[0]
        except OSError:
            key = None
    return key or None


def get_client():
    global _client
    if os.environ.get("VIRALYST_OFFLINE") == "1":
        raise VLMUnavailable("offline mode")
    with _lock:
        if _client is None:
            key = _api_key()
            if not key:
                raise VLMUnavailable("OPENAI_API_KEY not set")
            from openai import OpenAI
            _client = OpenAI(api_key=key, timeout=float(get_config().get("vlm.timeout_s")),
                             max_retries=int(get_config().get("vlm.client_retries")))
        return _client


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def image_part(path: str | Path) -> dict:
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}


def call_json(prompt: str, images: list[tuple[str, str]], schema: type[T]) -> tuple[T, str]:
    """Send prompt + labelled images; return (validated object, model used).

    ``images`` is a list of (label, path). Tries the configured model, then fallbacks.
    Raises VLMUnavailable if every model fails.
    """
    cfg = get_config()
    client = get_client()
    content: list[dict] = [{"type": "input_text", "text": prompt}]
    for label, path in images:
        content.append({"type": "input_text", "text": label})
        content.append(image_part(path))
    models = [str(cfg.get("vlm.model"))] + list(cfg.get("vlm.fallback_models"))
    retries = int(cfg.get("vlm.max_retries"))
    errors: list[str] = []
    for model in models:
        for _ in range(retries + 1):
            try:
                resp = client.responses.create(model=model, input=[{"role": "user", "content": content}],
                                               text={"format": {"type": "json_object"}})
                return schema.model_validate(json.loads(resp.output_text)), model
            except (json.JSONDecodeError, ValidationError) as e:
                errors.append(f"{model}: invalid JSON ({type(e).__name__})")
            except Exception as e:  # noqa: BLE001 - network/API errors -> next model
                errors.append(f"{model}: {type(e).__name__}: {str(e)[:120]}")
                break
    raise VLMUnavailable("; ".join(errors))
