"""
Тесты для модуля голосового ввода.

Модуль содержит как автоматические тесты (для проверки структуры),
так и интерактивные тесты для ручной проверки функциональности.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch, Mock

# Добавим src в путь для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestVoiceInputStructure(unittest.TestCase):
    """Автоматические тесты структуры модуля."""
    
    def test_module_imports(self):
        """Проверка, что модуль импортируется."""
        try:
            import voice_input
            self.assertTrue(hasattr(voice_input, 'VoiceInputHandler'))
        except ImportError as e:
            self.fail(f"Не удалось импортировать voice_input: {e}")
    
    def test_handler_class_exists(self):
        """Проверка существования класса VoiceInputHandler."""
        import voice_input
        self.assertTrue(hasattr(voice_input, 'VoiceInputHandler'))
        
        # Проверяем, что это класс
        self.assertTrue(callable(voice_input.VoiceInputHandler))
    
    def test_handler_has_required_methods(self):
        """Проверка наличия необходимых методов."""
        import voice_input
        
        handler = voice_input.VoiceInputHandler()
        self.assertTrue(hasattr(handler, 'listen_until_enter'))
        self.assertTrue(callable(handler.listen_until_enter))
    
    def test_handler_initialization(self):
        """Проверка инициализации обработчика."""
        import voice_input
        
        # Инициализация с языком по умолчанию
        handler1 = voice_input.VoiceInputHandler()
        self.assertEqual(handler1.language, "ru-RU")
        
        # Инициализация с другим языком
        handler2 = voice_input.VoiceInputHandler(language="en-US")
        self.assertEqual(handler2.language, "en-US")


class TestMainIntegration(unittest.TestCase):
    """Тесты интеграции с main.py."""
    
    def test_main_has_voice_command(self):
        """Проверка наличия обработчика /voice в main.py."""
        import main
        
        # Создаем минимальную конфигурацию для инициализации
        with patch('main.load_config') as mock_config:
            mock_config.return_value = {
                'llm_provider': 'local',
                'servers': {},
                'chat_model': {
                    'host': 'localhost',
                    'port': 11434,
                    'model_name': 'test',
                    'temperature': 0.7
                },
                'embedding_model': {
                    'host': 'localhost',
                    'port': 11434,
                    'model_name': 'test',
                    'endpoint': '/api/embeddings'
                }
            }
            
            try:
                assistant = main.SupportAssistant()
                self.assertTrue(hasattr(assistant, '_handle_voice_input'))
                self.assertTrue(callable(assistant._handle_voice_input))
            except Exception as e:
                # Если не удалось создать ассистента из-за отсутствия конфигов,
                # проверим хотя бы наличие класса и методов
                self.assertTrue(hasattr(main.SupportAssistant, '_handle_voice_input'))


def manual_test_voice_input():
    """
    Интерактивный тест голосового ввода.
    
    Запустите этот тест вручную для проверки:
        python test_voice_input.py manual
    """
    print("\n" + "="*60)
    print("ИНТЕРАКТИВНЫЙ ТЕСТ ГОЛОСОВОГО ВВОДА")
    print("="*60)
    print("\nЭтот тест проверит работу голосового ввода.")
    print("Вам нужно будет говорить в микрофон.\n")
    
    response = input("Продолжить? (y/n): ")
    if response.lower() != 'y':
        print("Тест отменен.")
        return
    
    try:
        import voice_input
        
        print("\n1. Инициализация обработчика...")
        handler = voice_input.VoiceInputHandler(language="ru-RU")
        print("✓ Обработчик создан")
        
        print("\n2. Начинаем прослушивание...")
        print("   Скажите что-нибудь по-русски и нажмите Enter")
        
        result = handler.listen_until_enter()
        
        if result:
            print(f"\n✓ УСПЕХ! Распознано: {result}")
        else:
            print("\n✗ Не удалось распознать речь")
            
    except ImportError as e:
        print(f"\n✗ Ошибка импорта: {e}")
        print("Установите зависимости: pip install SpeechRecognition PyAudio")
    except Exception as e:
        print(f"\n✗ Ошибка: {e}")


def manual_test_full_integration():
    """
    Интерактивный тест полной интеграции с main.py.
    
    Запустите этот тест вручную для проверки:
        python test_voice_input.py integration
    """
    print("\n" + "="*60)
    print("ИНТЕРАКТИВНЫЙ ТЕСТ ПОЛНОЙ ИНТЕГРАЦИИ")
    print("="*60)
    print("\nЭтот тест запустит приложение и протестирует команду /voice")
    print("Убедитесь, что все конфигурационные файлы настроены.\n")
    
    response = input("Продолжить? (y/n): ")
    if response.lower() != 'y':
        print("Тест отменен.")
        return
    
    print("\nЗапуск приложения...")
    print("Введите команду: /voice")
    print("Затем скажите что-нибудь и нажмите Enter\n")
    
    try:
        import main
        main.main()
    except KeyboardInterrupt:
        print("\n\nТест прерван пользователем.")
    except Exception as e:
        print(f"\n✗ Ошибка: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "manual":
            manual_test_voice_input()
        elif sys.argv[1] == "integration":
            manual_test_full_integration()
        else:
            print("Использование:")
            print("  python test_voice_input.py          # Автоматические тесты")
            print("  python test_voice_input.py manual   # Тест голосового ввода")
            print("  python test_voice_input.py integration  # Тест интеграции")
    else:
        # Запуск автоматических тестов
        unittest.main()
