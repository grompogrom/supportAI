"""
Тесты для LLM клиента.
"""

import sys
import os
import yaml

# Добавляем путь к src для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_client import LLMClient, AuthenticationError, RateLimitError, APIError


def load_api_key():
    """Загружает API ключ из конфига."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'api_keys.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['perplexity']['api_key']


# API ключ для тестирования (загружается из config/api_keys.yaml)
API_KEY = load_api_key()


def test_initialization():
    """Тест инициализации клиента."""
    print("=" * 50)
    print("Тест 1: Инициализация клиента")
    print("=" * 50)
    
    client = LLMClient(api_key=API_KEY, system_prompt="Ты полезный ассистент")
    
    history = client.get_messages_history()
    assert len(history) == 1, f"Ожидалось 1 сообщение, получено {len(history)}"
    assert history[0]["role"] == "system", "Первое сообщение должно быть системным"
    assert history[0]["content"] == "Ты полезный ассистент", "Неверный системный промпт"
    
    print("✓ Инициализация успешна")
    print(f"  История: {history}")
    print()
    return True


def test_send_message():
    """Тест отправки сообщения."""
    print("=" * 50)
    print("Тест 2: Отправка сообщения")
    print("=" * 50)
    
    client = LLMClient(api_key=API_KEY, system_prompt="Ты полезный ассистент. Отвечай кратко.")
    
    try:
        response = client.send_message("Скажи 'привет' одним словом")
        print(f"✓ Ответ получен: {response[:100]}...")
        
        history = client.get_messages_history()
        assert len(history) == 3, f"Ожидалось 3 сообщения, получено {len(history)}"
        assert history[1]["role"] == "user", "Второе сообщение должно быть от пользователя"
        assert history[2]["role"] == "assistant", "Третье сообщение должно быть от ассистента"
        
        print(f"  История содержит {len(history)} сообщений")
        print()
        return True
    except Exception as e:
        print(f"✗ Ошибка: {type(e).__name__}: {e}")
        print()
        return False


def test_clear_history():
    """Тест очистки истории."""
    print("=" * 50)
    print("Тест 3: Очистка истории")
    print("=" * 50)
    
    client = LLMClient(api_key=API_KEY, system_prompt="Системный промпт")
    
    # Добавим сообщение вручную для теста (без API вызова)
    client._messages.append({"role": "user", "content": "тест"})
    client._messages.append({"role": "assistant", "content": "ответ"})
    
    assert len(client.get_messages_history()) == 3
    
    client.clear_history()
    
    history = client.get_messages_history()
    assert len(history) == 1, f"После очистки должно быть 1 сообщение, получено {len(history)}"
    assert history[0]["role"] == "system", "Системный промпт должен сохраниться"
    assert history[0]["content"] == "Системный промпт"
    
    print("✓ Очистка истории работает корректно")
    print()
    return True


def test_set_system_prompt():
    """Тест изменения системного промпта."""
    print("=" * 50)
    print("Тест 4: Изменение системного промпта")
    print("=" * 50)
    
    client = LLMClient(api_key=API_KEY, system_prompt="Старый промпт")
    
    client.set_system_prompt("Новый промпт")
    
    history = client.get_messages_history()
    assert history[0]["content"] == "Новый промпт", "Системный промпт не обновился"
    
    print("✓ Изменение системного промпта работает")
    print()
    return True


def test_get_messages_history_returns_copy():
    """Тест что get_messages_history возвращает копию."""
    print("=" * 50)
    print("Тест 5: Возврат копии истории")
    print("=" * 50)
    
    client = LLMClient(api_key=API_KEY, system_prompt="Тест")
    
    history1 = client.get_messages_history()
    history1.append({"role": "user", "content": "хак"})
    
    history2 = client.get_messages_history()
    assert len(history2) == 1, "Изменение копии не должно влиять на оригинал"
    
    print("✓ get_messages_history возвращает копию")
    print()
    return True


def test_invalid_api_key():
    """Тест с неверным API ключом."""
    print("=" * 50)
    print("Тест 6: Неверный API ключ")
    print("=" * 50)
    
    client = LLMClient(api_key="invalid-key", system_prompt="Тест")
    
    try:
        client.send_message("Привет")
        print("✗ Ожидалась ошибка AuthenticationError")
        return False
    except AuthenticationError as e:
        print(f"✓ Получена ожидаемая ошибка: {e}")
        print()
        return True
    except Exception as e:
        print(f"✗ Получена неожиданная ошибка: {type(e).__name__}: {e}")
        print()
        return False


def test_send_tool_result():
    """Тест отправки результата инструмента."""
    print("=" * 50)
    print("Тест 7: Отправка результата инструмента")
    print("=" * 50)
    
    client = LLMClient(api_key=API_KEY, system_prompt="Ты ассистент. Отвечай кратко.")
    
    try:
        # Сначала отправим сообщение
        client.send_message("Я сейчас отправлю тебе результат инструмента")
        
        # Теперь отправим результат инструмента
        tool_result = {"status": "success", "data": [1, 2, 3]}
        response = client.send_tool_result("test_tool", tool_result)
        
        print(f"✓ Ответ на tool result: {response[:100]}...")
        
        history = client.get_messages_history()
        # system + user + assistant + user (tool result) + assistant
        assert len(history) == 5, f"Ожидалось 5 сообщений, получено {len(history)}"
        
        print()
        return True
    except Exception as e:
        print(f"✗ Ошибка: {type(e).__name__}: {e}")
        print()
        return False


def run_all_tests():
    """Запуск всех тестов."""
    print("\n" + "=" * 50)
    print("ЗАПУСК ТЕСТОВ LLM CLIENT")
    print("=" * 50 + "\n")
    
    tests = [
        ("Инициализация", test_initialization),
        ("Очистка истории", test_clear_history),
        ("Изменение системного промпта", test_set_system_prompt),
        ("Возврат копии истории", test_get_messages_history_returns_copy),
        ("Неверный API ключ", test_invalid_api_key),
        ("Отправка сообщения", test_send_message),
        ("Отправка результата инструмента", test_send_tool_result),
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
