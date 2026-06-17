import logging
import os
import re

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, retry_if_exception_type

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

_cfg_logger = logging.getLogger(__name__)


def _is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "quota" in msg or "resource_exhausted" in msg


def _parse_retry_seconds(exc: BaseException) -> float:
    """Extract 'retry in Xs' from a 429 error message, defaulting to 25s."""
    match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", str(exc), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2  # add 2s buffer
    return 25.0


def _smart_wait(retry_state) -> float:
    """Return wait seconds: API-suggested time for 429s, exponential backoff otherwise."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc and _is_rate_limit_error(exc):
        wait = _parse_retry_seconds(exc)
        _cfg_logger.warning("[llm_retry] Rate limit (429) — waiting %.1fs as requested by API", wait)
        return wait
    attempt = retry_state.attempt_number
    # Increased wait for non-429 or generic errors to prevent hammering
    return min(5 * (2 ** attempt), 60)


llm_retry = retry(
    retry=retry_if_exception_type(Exception),
    wait=_smart_wait,
    stop=stop_after_attempt(6),
    reraise=True,
)


def get_llm_client():
    """Return a provider-specific LLM callable.

    Returns a dict with keys:
        - 'provider': str
        - 'model': str
        - 'call': callable(system_prompt: str, user_prompt: str) -> str

    The 'call' function is already wrapped with tenacity retry/backoff.
    """
    if LLM_PROVIDER == "gemini":
        return _build_gemini_client()
    elif LLM_PROVIDER == "openai":
        return _build_openai_client()
    elif LLM_PROVIDER in ("claude", "anthropic"):
        return _build_anthropic_client()
    elif LLM_PROVIDER == "groq":
        return _build_groq_client()
    elif LLM_PROVIDER == "openrouter":
        return _build_openrouter_client()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'. Set to 'gemini', 'openai', 'claude', 'groq', or 'openrouter'.")


# ---------- Gemini ----------

def _build_gemini_client():
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_API_KEY)

    @llm_retry
    def _call(system_prompt: str, user_prompt: str) -> str:
        try:
            response = client.models.generate_content(
                model=LLM_MODEL_NAME,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                ),
            )
            return response.text
        except Exception as exc:
            if _is_rate_limit_error(exc):
                raise  # tenacity will retry
            raise

    return {"provider": "gemini", "model": LLM_MODEL_NAME, "call": _call}


# ---------- OpenAI ----------

def _build_openai_client():
    from openai import OpenAI, RateLimitError
    client = OpenAI(api_key=OPENAI_API_KEY)

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=_smart_wait,
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _call(system_prompt: str, user_prompt: str) -> str:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    return {"provider": "openai", "model": LLM_MODEL_NAME, "call": _call}


# ---------- Groq (OpenAI-compatible) ----------

def _build_groq_client():
    from openai import OpenAI
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

    @llm_retry
    def _call(system_prompt: str, user_prompt: str) -> str:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    return {"provider": "groq", "model": LLM_MODEL_NAME, "call": _call}


# ---------- OpenRouter (OpenAI-compatible) ----------

def _build_openrouter_client():
    from openai import OpenAI
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    @llm_retry
    def _call(system_prompt: str, user_prompt: str) -> str:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            extra_headers={
                "HTTP-Referer": "https://aegis-clinical.ai",
                "X-Title": "AEGIS Clinical Safety Assistant",
            },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    return {"provider": "openrouter", "model": LLM_MODEL_NAME, "call": _call}


# ---------- Anthropic / Claude ----------

def _build_anthropic_client():
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    @llm_retry
    def _call(system_prompt: str, user_prompt: str) -> str:
        try:
            message = client.messages.create(
                model=LLM_MODEL_NAME,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return message.content[0].text
        except Exception as exc:
            if _is_rate_limit_error(exc):
                raise
            raise

    return {"provider": "claude", "model": LLM_MODEL_NAME, "call": _call}


# ---------- Vision helper (multimodal) ----------

def call_vision_llm(image_bytes: bytes, prompt: str) -> str:
    """Send an image + prompt to the vision-capable model and return extracted text."""

    @llm_retry
    def _gemini_vision():
        from google import genai
        from google.genai import types
        import PIL.Image
        import io
        client = genai.Client(api_key=GEMINI_API_KEY)
        img = PIL.Image.open(io.BytesIO(image_bytes))
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, img],
        )
        return response.text

    @llm_retry
    def _openai_vision():
        import base64
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
        )
        return response.choices[0].message.content

    if LLM_PROVIDER == "gemini":
        return _gemini_vision()
    elif LLM_PROVIDER == "openai":
        return _openai_vision()
    else:
        return _gemini_vision()
