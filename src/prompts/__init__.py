"""
Модуль системных промптов для LLM.

Содержит:
    - SYSTEM_PROMPT: Основной системный промпт для ассистента поддержки
    - get_system_prompt(): Функция для получения промпта с динамическими параметрами
"""

from .system_prompt import (
    SYSTEM_PROMPT,
    TOOL_DEFINITIONS,
    get_system_prompt,
    format_tools_description,
    get_tool_definitions,
)

__all__ = [
    "SYSTEM_PROMPT",
    "TOOL_DEFINITIONS",
    "get_system_prompt",
    "format_tools_description",
    "get_tool_definitions",
]
