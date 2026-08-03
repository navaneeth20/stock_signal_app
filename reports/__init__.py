"""
reports package initialization.
"""

from reports.institutional_llm import (
    INSTITUTIONAL_PROMPTS,
    call_gemini_api,
    call_openai_api,
    generate_fallback_institutional_report,
)

__all__ = [
    "INSTITUTIONAL_PROMPTS",
    "call_gemini_api",
    "call_openai_api",
    "generate_fallback_institutional_report",
]
