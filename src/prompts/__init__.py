"""
Модуль системных промптов для LLM.

Содержит:
    - SYSTEM_PROMPT: Основной системный промпт для AI менеджера проекта
    - get_system_prompt(): Функция для получения промпта с динамическими параметрами
    - format_tools_description(): Функция для форматирования описаний инструментов
"""

from .system_prompt import (
    SYSTEM_PROMPT,
    get_system_prompt,
    format_tools_description,
)

__all__ = [
    "SYSTEM_PROMPT",
    "get_system_prompt",
    "format_tools_description",
]
