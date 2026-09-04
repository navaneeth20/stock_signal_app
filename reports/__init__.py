"""
reports package initialization.
"""

from reports.institutional_llm import (
    INSTITUTIONAL_PROMPTS,
    NUMERIC_PROMPTS,
    RESEARCH_DISCLAIMER,
    build_grounded_prompt,
    call_gemini_api,
    call_openai_api,
    generate_fallback_institutional_report,
)

__all__ = [
    "INSTITUTIONAL_PROMPTS",
    "NUMERIC_PROMPTS",
    "RESEARCH_DISCLAIMER",
    "build_grounded_prompt",
    "call_gemini_api",
    "call_openai_api",
    "generate_fallback_institutional_report",
]
