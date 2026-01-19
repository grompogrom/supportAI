"""
Тесты для модуля системного промпта.
"""

import sys
import os

# Добавляем путь к src для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prompts.system_prompt import (
    SYSTEM_PROMPT,
    get_system_prompt,
    format_tools_description,
)


def test_get_system_prompt_without_args():
    """Тест get_system_prompt без аргументов."""
    print("=" * 50)
    print("Тест 1: get_system_prompt() без аргументов")
    print("=" * 50)
    
    result = get_system_prompt()
    
    assert result == SYSTEM_PROMPT, "Должен вернуть SYSTEM_PROMPT без изменений"
    assert "AI менеджер проекта" in result, "Промпт должен содержать описание роли"
    assert "create_ticket" in result, "Промпт должен содержать инструмент create_ticket"
    assert "recommend_tasks" in result, "Промпт должен содержать инструмент recommend_tasks"
    
    print("✓ get_system_prompt() возвращает SYSTEM_PROMPT без изменений")
    print()
    return True


def test_get_system_prompt_with_custom_tools():
    """Тест get_system_prompt с кастомными инструментами."""
    print("=" * 50)
    print("Тест 2: get_system_prompt() с кастомными инструментами")
    print("=" * 50)
    
    custom_tools = [
        {
            "name": "custom_tool",
            "description": "Кастомный инструмент для теста",
            "parameters": {
                "param1": {
                    "type": "string",
                    "required": True,
                    "description": "Первый параметр"
                }
            }
        }
    ]
    
    result = get_system_prompt(custom_tools)
    
    assert result != SYSTEM_PROMPT, "Должен вернуть измененный промпт"
    assert "custom_tool" in result, "Промпт должен содержать кастомный инструмент"
    assert "Кастомный инструмент для теста" in result, "Промпт должен содержать описание"
    assert "## ДОСТУПНЫЕ ИНСТРУМЕНТЫ:" in result, "Промпт должен содержать секцию инструментов"
    assert "## ФОРМАТ ВЫЗОВА ИНСТРУМЕНТА:" in result, "Промпт должен содержать формат вызова"
    
    print("✓ get_system_prompt(tools) генерирует промпт с кастомными инструментами")
    print(f"  Промпт содержит {len(result)} символов")
    print()
    return True


def test_format_tools_description_basic():
    """Тест базового форматирования инструментов."""
    print("=" * 50)
    print("Тест 3: format_tools_description базовый")
    print("=" * 50)
    
    tools = [
        {
            "name": "test_tool",
            "description": "Тестовый инструмент",
            "parameters": {
                "required_param": {
                    "type": "string",
                    "required": True,
                    "description": "Обязательный параметр"
                },
                "optional_param": {
                    "type": "integer",
                    "required": False,
                    "description": "Опциональный параметр"
                }
            }
        }
    ]
    
    result = format_tools_description(tools)
    
    assert "### 1. test_tool" in result, "Должен содержать заголовок с номером"
    assert "**Описание:** Тестовый инструмент" in result, "Должен содержать описание"
    assert "**Параметры:**" in result, "Должен содержать секцию параметров"
    assert "required_param (string, (required)): Обязательный параметр" in result
    assert "optional_param (integer, (optional)): Опциональный параметр" in result
    
    print("✓ format_tools_description форматирует базовые параметры")
    print(f"  Результат:\n{result}")
    print()
    return True


def test_format_tools_description_with_defaults():
    """Тест форматирования параметров со значениями по умолчанию."""
    print("=" * 50)
    print("Тест 4: format_tools_description с default значениями")
    print("=" * 50)
    
    tools = [
        {
            "name": "tool_with_defaults",
            "description": "Инструмент с дефолтами",
            "parameters": {
                "priority": {
                    "type": "string",
                    "required": False,
                    "description": "Приоритет",
                    "default": "medium"
                },
                "count": {
                    "type": "integer",
                    "required": False,
                    "description": "Количество",
                    "default": 10
                }
            }
        }
    ]
    
    result = format_tools_description(tools)
    
    assert "[по умолчанию: medium]" in result, "Должен содержать дефолтное значение medium"
    assert "[по умолчанию: 10]" in result, "Должен содержать дефолтное значение 10"
    
    print("✓ format_tools_description корректно отображает default значения")
    print()
    return True


def test_format_tools_description_multiple_tools():
    """Тест форматирования нескольких инструментов."""
    print("=" * 50)
    print("Тест 5: format_tools_description с несколькими инструментами")
    print("=" * 50)
    
    tools = [
        {
            "name": "tool_one",
            "description": "Первый инструмент",
            "parameters": {}
        },
        {
            "name": "tool_two",
            "description": "Второй инструмент",
            "parameters": {}
        },
        {
            "name": "tool_three",
            "description": "Третий инструмент",
            "parameters": {}
        }
    ]
    
    result = format_tools_description(tools)
    
    assert "### 1. tool_one" in result, "Должен содержать первый инструмент"
    assert "### 2. tool_two" in result, "Должен содержать второй инструмент"
    assert "### 3. tool_three" in result, "Должен содержать третий инструмент"
    
    print("✓ format_tools_description нумерует инструменты корректно")
    print()
    return True


def test_format_tools_description_empty_parameters():
    """Тест форматирования инструмента без параметров."""
    print("=" * 50)
    print("Тест 6: format_tools_description без параметров")
    print("=" * 50)
    
    tools = [
        {
            "name": "no_params_tool",
            "description": "Инструмент без параметров",
            "parameters": {}
        }
    ]
    
    result = format_tools_description(tools)
    
    assert "### 1. no_params_tool" in result
    assert "**Описание:** Инструмент без параметров" in result
    assert "**Параметры:**" in result
    # После "**Параметры:**" не должно быть параметров, только пустая строка
    
    print("✓ format_tools_description обрабатывает пустые параметры")
    print()
    return True


def test_format_tools_description_missing_type():
    """Тест обработки параметра без указанного типа."""
    print("=" * 50)
    print("Тест 7: Параметр без указанного типа")
    print("=" * 50)
    
    tools = [
        {
            "name": "tool_no_type",
            "description": "Инструмент с параметром без типа",
            "parameters": {
                "param": {
                    "required": True,
                    "description": "Параметр без типа"
                }
            }
        }
    ]
    
    result = format_tools_description(tools)
    
    # Должен использовать 'string' по умолчанию
    assert "param (string, (required))" in result, "Должен использовать string по умолчанию"
    
    print("✓ format_tools_description использует 'string' по умолчанию для типа")
    print()
    return True


def run_all_tests():
    """Запуск всех тестов."""
    print("\n" + "=" * 50)
    print("ЗАПУСК ТЕСТОВ SYSTEM PROMPT")
    print("=" * 50 + "\n")
    
    tests = [
        ("get_system_prompt без аргументов", test_get_system_prompt_without_args),
        ("get_system_prompt с кастомными инструментами", test_get_system_prompt_with_custom_tools),
        ("format_tools_description базовый", test_format_tools_description_basic),
        ("format_tools_description с default", test_format_tools_description_with_defaults),
        ("format_tools_description несколько инструментов", test_format_tools_description_multiple_tools),
        ("format_tools_description без параметров", test_format_tools_description_empty_parameters),
        ("Параметр без типа", test_format_tools_description_missing_type),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Тест '{name}' упал с ошибкой: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ ТЕСТОВ")
    print("=" * 50)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nИтого: {passed}/{total} тестов пройдено")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
