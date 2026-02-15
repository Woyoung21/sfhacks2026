import os
import re

from google.genai import Client
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("GOOGLE_API_KEY")
if not key:
    print("API Key not found. do you have GOOGLE_API_KEY in .env?")

client = Client(api_key=key)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "250"))
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.4"))


def _normalize_response(text: str) -> str:
    """Convert markdown-heavy output into clean plain language."""
    out = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    out = out.replace("```", "")
    out = re.sub(r"^\s{0,3}#{1,6}\s*", "", out, flags=re.MULTILINE)
    out = out.replace("**", "").replace("__", "")
    out = out.replace("`", "")
    out = out.replace("$", "")
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def make_call(prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                temperature=GEMINI_TEMPERATURE,
                system_instruction=(
                    "Reply in clear natural language. "
                    "Avoid markdown syntax, code fences, and symbolic notation unless requested."
                ),
                response_mime_type="text/plain",
            ),
        )

        return _normalize_response(response.text or "")
    
    except Exception as e:
        return f"Error making API call: {str(e)}"
