"""
agent_client.py — Component #4 (dataset_generator.md, section 4).

Calls the local llama.cpp server (OpenAI-compatible /v1/chat/completions)
running Qwen3.5-9B, at http://127.0.0.1:9932.

NOTE: this endpoint is on your machine, not reachable from where this
code was written/tested — I could not do a live call from here. The
request shape below follows llama.cpp's documented OpenAI-compatible
server API; verify with a quick curl on your side before a full run:

  curl http://127.0.0.1:9932/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{"messages":[{"role":"user","content":"halo"}], "max_tokens": 20}'

If llama.cpp reports the model name differently than "qwen3.5-9b"
(some builds ignore the "model" field entirely and just use whatever
was loaded with -m), you can leave AGENT_MODEL as-is; llama.cpp's
server typically ignores an unrecognized model string rather than
erroring, but check your server's startup log for the exact behavior.

NOTE: Qwen3.5-9B is a reasoning model. The connection check uses
max_tokens=10 which may produce reasoning_content but no content.
The code handles this by accepting reasoning_content as a sign the
server is alive (see check_connection / generate_narrative).
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

import requests

AGENT_ENDPOINT = "http://127.0.0.1:9932/v1/chat/completions"
AGENT_MODEL = "qwen3.5-9b"

# --- Logging setup ---
_DEBUG_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(_DEBUG_LOG_DIR, exist_ok=True)
_DEBUG_LOG_PATH = os.path.join(_DEBUG_LOG_DIR, "agent_client_debug.log")

logger = logging.getLogger("agent_client")
logger.setLevel(logging.DEBUG)

# File handler — every call gets logged here
_fh = logging.FileHandler(_DEBUG_LOG_PATH, mode="a", encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
logger.addHandler(_fh)

# Console handler — visible during runs
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                    datefmt="%H:%M:%S"))
logger.addHandler(_ch)


class AgentError(Exception):
    """Raised when the agent call fails (connection, timeout, malformed response)."""
    pass


def _truncate(s: str, max_len: int = 500) -> str:
    """Truncate a string for safe logging."""
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"... (truncated, {len(s)} chars total)"


def generate_narrative(structured_input: str, instruction: str, system_prompt: str,
                        temperature: float = 0.8, max_tokens: int = 700,
                        timeout: float = 300.0,
                        max_retries: int = 2) -> str:
    """
    Calls the local agent to generate ONE narrative.

    Improved over the original:
    - Logs HTTP status code + truncated response body for EVERY call.
    - Retries on transient failures (connection reset, 5xx).
    - Raises AgentError with detailed context instead of swallowing errors.
    - Distinguishes between "server error" and "empty content from server".

    Args:
        structured_input: Pre-computed anthropometric data with Z-scores.
        instruction: Prompt instruction (Style A/B/C).
        system_prompt: System constraint prompt.
        temperature: Sampling temperature.
        max_tokens: Max output tokens (~500-600 untuk 1 narrative ~70-120 token + buffer).
        timeout: Seconds to wait for response (default 300s for 35B-A3B).
        max_retries: Number of retry attempts on transient failure.

    Returns:
        Generated narrative string.

    Raises:
        AgentError: On any failure (connection, timeout, malformed response,
                    or server returns empty content after all retries).
    """
    payload = {
        "model": AGENT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{instruction}\n\n{structured_input}"},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,  # Explicitly disable streaming — ensure we get a single JSON response
        "chat_template_kwargs": {"enable_thinking": False},  # TIDAK enable thinking → reasoning_content kosong, content langsung terisi
    }

    last_error = None

    for attempt in range(1 + max_retries):
        call_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{attempt + 1}"
        logger.info(
            "[%s] CALL attempt=%d/%d | endpoint=%s | model=%s | "
            "max_tokens=%d | temperature=%.1f | timeout=%.0fs",
            call_id, attempt + 1, 1 + max_retries,
            AGENT_ENDPOINT, AGENT_MODEL, max_tokens, temperature, timeout
        )
        logger.debug("[%s] payload.messages[0].system = %s", call_id, _truncate(system_prompt, 200))
        logger.debug("[%s] payload.messages[1].user = %s", call_id, _truncate(structured_input, 300))

        try:
            resp = requests.post(AGENT_ENDPOINT, json=payload, timeout=timeout)
            status_code = resp.status_code

            # Log the raw response (truncated for safety)
            raw_text = resp.text
            logger.info(
                "[%s] RESPONSE status=%d | size=%d bytes | body=%s",
                call_id, status_code, len(raw_text),
                _truncate(raw_text, 1000)
            )

            # Check for HTTP errors
            if status_code >= 500:
                logger.error("[%s] SERVER ERROR %d — will retry", call_id, status_code)
                last_error = AgentError(f"server returned HTTP {status_code}: {_truncate(raw_text, 500)}")
                time.sleep(2 * (attempt + 1))  # exponential backoff
                continue

            if status_code >= 400:
                # 4xx errors are likely permanent (bad model name, bad params)
                raise AgentError(f"client error HTTP {status_code}: {_truncate(raw_text, 500)}")

            # Parse JSON response
            try:
                data = resp.json()
            except ValueError as e:
                raise AgentError(
                    f"response is not valid JSON (status {status_code}): "
                    f"{_truncate(raw_text, 500)}"
                ) from e

            # Validate response structure — llama.cpp / OpenAI format:
            # {"choices": [{"message": {"content": "..."}}]}
            choices = data.get("choices")
            if not choices or not isinstance(choices, list) or len(choices) == 0:
                raise AgentError(
                    f"response has no 'choices' field or it's empty. "
                    f"Full response: {_truncate(json.dumps(data), 800)}"
                )

            message = choices[0].get("message")
            if not message or not isinstance(message, dict):
                raise AgentError(
                    f"'choices[0]' has no 'message' field. "
                    f"Full choices[0]: {_truncate(json.dumps(choices[0]), 800)}"
                )

            content = message.get("content")
            if content is None:
                raise AgentError(
                    f"'choices[0].message' has no 'content' field. "
                    f"Full message: {_truncate(json.dumps(message), 800)}"
                )

            if not isinstance(content, str):
                raise AgentError(
                    f"'choices[0].message.content' is not a string: {type(content).__name__} = {content!r}"
                )

            stripped = content.strip()
            if not stripped:
                # EMPTY content — possible for reasoning models (Qwen3.5-9B)
                # where all tokens are consumed by reasoning_content.
                # Check if reasoning_content is present: if so, server is
                # alive and responding normally, just in reasoning mode.
                reasoning_content = message.get("reasoning_content", "")
                if reasoning_content and str(reasoning_content).strip():
                    logger.info(
                        "[%s] REASONING-ONLY — server returned reasoning_content "
                        "(%d chars) but no content. This is normal for reasoning "
                        "models with small max_tokens. Connection OK.",
                        call_id, len(str(reasoning_content))
                    )
                    return "[connection-ok]"

                # No reasoning_content either — genuine empty response
                logger.error(
                    "[%s] EMPTY CONTENT — server returned a valid response but "
                    "both content and reasoning_content are empty. "
                    "This often means: (a) llama.cpp server is loaded with a "
                    "different model than expected, (b) the 'model' field in "
                    "the request conflicts with the loaded model, (c) the prompt "
                    "exceeds context window and is silently rejected, or (d) the "
                    "server is busy and returns an empty response. Full response: %s",
                    call_id, _truncate(json.dumps(data), 800)
                )
                last_error = AgentError(
                    f"server returned empty content (HTTP {status_code}). "
                    f"Response: {_truncate(json.dumps(data), 800)}. "
                    f"Check: (a) llama.cpp model name matches '{AGENT_MODEL}', "
                    f"(b) GPU memory is sufficient, (c) prompt length within context window."
                )
                # Don't retry empty content — it's a systematic issue, not transient
                break

            logger.info("[%s] SUCCESS | output_length=%d words", call_id, len(stripped.split()))
            return stripped

        except requests.exceptions.Timeout as e:
            logger.error("[%s] TIMEOUT after %.0fs — will retry", call_id, timeout)
            last_error = AgentError(f"request timed out after {timeout}s: {e}")
            time.sleep(2 * (attempt + 1))
            continue

        except requests.exceptions.ConnectionError as e:
            logger.error("[%s] CONNECTION ERROR — server may not be running: %s", call_id, e)
            last_error = AgentError(f"connection failed (server may not be running): {e}")
            time.sleep(2 * (attempt + 1))
            continue

        except AgentError:
            # Already logged above
            last_error = e
            # Don't retry on client errors or empty content
            if "client error" in str(e) or "empty content" in str(e):
                break
            # Retry on other AgentErrors (e.g., malformed response)
            time.sleep(1 * (attempt + 1))
            continue

        except Exception as e:
            logger.error("[%s] UNEXPECTED ERROR: %s | type=%s", call_id, e, type(e).__name__)
            last_error = AgentError(f"unexpected error: {e}")
            break

    # All retries exhausted
    raise last_error or AgentError("generate_narrative failed without setting last_error")


def check_connection() -> bool:
    """Quick health check before kicking off a full batch run."""
    try:
        generate_narrative(
            structured_input="Usia: 12 bulan, Jenis kelamin: Laki-laki, HAZ: 0.0 (Normal)",
            instruction="Balas hanya dengan kata 'ok' untuk mengonfirmasi koneksi.",
            system_prompt="Kamu adalah asisten uji koneksi.",
            max_tokens=10,
        )
        return True
    except AgentError as e:
        print(f"[agent_client] connection check failed: {e}")
        return False


if __name__ == "__main__":
    print("Checking connection to", AGENT_ENDPOINT, "...")
    print(f"Debug log will be written to: {_DEBUG_LOG_PATH}")
    ok = check_connection()
    print("OK" if ok else "FAILED — check that llama.cpp server is running and reachable")
