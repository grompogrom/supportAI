"""
Примеры использования LLM клиентов.

Демонстрирует работу с Perplexity API и локальными моделями через Ollama.
"""

import sys
import os

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_client import PerplexityClient, LocalLLMClient


def example_perplexity():
    """Пример использования Perplexity API."""
    print("=" * 60)
    print("Пример 1: Perplexity API")
    print("=" * 60)
    
    # Инициализация клиента
    client = PerplexityClient(
        api_key="your-api-key-here",
        system_prompt="Ты полезный ассистент. Отвечай кратко и по делу."
    )
    
    # Отправка сообщения
    try:
        response = client.send_message("Привет! Расскажи о себе в 2-3 предложениях.")
        print(f"\nОтвет: {response}\n")
        
        # История сообщений
        history = client.get_messages_history()
        print(f"История содержит {len(history)} сообщений")
        
    except Exception as e:
        print(f"Ошибка: {e}")


def example_local_llm():
    """Пример использования локальной модели через Ollama."""
    print("=" * 60)
    print("Пример 2: Локальная модель (qwen3:8b)")
    print("=" * 60)
    
    # Инициализация клиента
    client = LocalLLMClient(
        host="localhost",
        port=11434,
        model_name="qwen3:8b",
        system_prompt="Ты полезный ассистент. Отвечай кратко и по делу.",
        temperature=0.7
    )
    
    # Проверка доступности модели
    print("Проверка доступности модели...")
    if not client.check_model_availability():
        print("❌ Модель недоступна!")
        print("Убедитесь, что Ollama запущен и модель загружена:")
        print("  ollama run qwen3:8b")
        return
    
    print("✅ Модель доступна\n")
    
    # Отправка сообщения
    try:
        print("Отправка сообщения (может занять 10-30 секунд)...")
        response = client.send_message("Привет! Расскажи о себе в 2-3 предложениях.")
        print(f"\nОтвет: {response}\n")
        
        # История сообщений
        history = client.get_messages_history()
        print(f"История содержит {len(history)} сообщений")
        
    except Exception as e:
        print(f"Ошибка: {e}")


def example_conversation():
    """Пример диалога с несколькими сообщениями."""
    print("=" * 60)
    print("Пример 3: Диалог с несколькими сообщениями")
    print("=" * 60)
    
    # Используем локальную модель (замените на PerplexityClient при необходимости)
    client = LocalLLMClient(
        host="localhost",
        port=11434,
        model_name="qwen3:8b",
        system_prompt="Ты математический ассистент. Решай задачи пошагово.",
        temperature=0.3  # Низкая температура для более детерминированных ответов
    )
    
    if not client.check_model_availability():
        print("❌ Модель недоступна!")
        return
    
    # Диалог
    messages = [
        "Сколько будет 15 + 27?",
        "А если умножить этот результат на 2?",
        "Спасибо!"
    ]
    
    for msg in messages:
        print(f"\n👤 User: {msg}")
        try:
            response = client.send_message(msg)
            print(f"🤖 Assistant: {response}")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            break
    
    # Итоговая история
    print(f"\n📊 Всего сообщений в истории: {len(client.get_messages_history())}")


def example_tool_result():
    """Пример отправки результата инструмента."""
    print("=" * 60)
    print("Пример 4: Отправка результата инструмента")
    print("=" * 60)
    
    client = LocalLLMClient(
        host="localhost",
        port=11434,
        model_name="qwen3:8b",
        system_prompt="Ты ассистент службы поддержки. Анализируй результаты инструментов.",
        temperature=0.7
    )
    
    if not client.check_model_availability():
        print("❌ Модель недоступна!")
        return
    
    # Первое сообщение
    print("\n👤 User: Найди информацию о тикете TKT-123")
    try:
        response = client.send_message("Найди информацию о тикете TKT-123")
        print(f"🤖 Assistant: {response}")
        
        # Симуляция результата инструмента
        tool_result = {
            "ticket_id": "TKT-123",
            "status": "open",
            "priority": "high",
            "title": "Проблема с авторизацией",
            "created_at": "2026-01-15T10:30:00Z"
        }
        
        print("\n🔧 Tool result: get_ticket")
        print(f"   {tool_result}")
        
        # Отправка результата
        response = client.send_tool_result("get_ticket", tool_result)
        print(f"\n🤖 Assistant: {response}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")


def example_clear_and_update():
    """Пример очистки истории и обновления промпта."""
    print("=" * 60)
    print("Пример 5: Управление историей и промптом")
    print("=" * 60)
    
    client = LocalLLMClient(
        host="localhost",
        port=11434,
        model_name="qwen3:8b",
        system_prompt="Ты дружелюбный ассистент.",
        temperature=0.7
    )
    
    if not client.check_model_availability():
        print("❌ Модель недоступна!")
        return
    
    # Первый диалог
    print("\n--- Первый диалог ---")
    try:
        response = client.send_message("Привет!")
        print(f"Ответ: {response[:100]}...")
        print(f"История: {len(client.get_messages_history())} сообщений")
        
        # Очистка истории
        print("\n🧹 Очищаем историю...")
        client.clear_history()
        print(f"История: {len(client.get_messages_history())} сообщений")
        
        # Изменение промпта
        print("\n✏️ Меняем системный промпт...")
        client.set_system_prompt("Ты строгий технический эксперт. Отвечай формально.")
        
        # Новый диалог
        print("\n--- Новый диалог ---")
        response = client.send_message("Привет!")
        print(f"Ответ: {response[:100]}...")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")


def main():
    """Главная функция с меню."""
    print("\n" + "=" * 60)
    print("ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ LLM КЛИЕНТОВ")
    print("=" * 60)
    
    examples = {
        "1": ("Perplexity API", example_perplexity),
        "2": ("Локальная модель", example_local_llm),
        "3": ("Диалог", example_conversation),
        "4": ("Tool result", example_tool_result),
        "5": ("Управление историей", example_clear_and_update),
    }
    
    print("\nВыберите пример:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  0. Запустить все примеры")
    print("  q. Выход")
    
    choice = input("\nВаш выбор: ").strip()
    
    if choice == "q":
        print("До свидания!")
        return
    
    if choice == "0":
        for name, func in examples.values():
            print("\n")
            func()
            input("\nНажмите Enter для продолжения...")
    elif choice in examples:
        examples[choice][1]()
    else:
        print("Неверный выбор!")


if __name__ == "__main__":
    main()
