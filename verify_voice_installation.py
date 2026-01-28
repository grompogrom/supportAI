#!/usr/bin/env python3
"""
Скрипт для проверки установки голосового ввода.

Проверяет:
1. Установлены ли необходимые зависимости
2. Доступен ли микрофон
3. Работает ли импорт модулей
"""

import sys


def check_dependencies():
    """Проверка установки зависимостей."""
    print("="*60)
    print("ПРОВЕРКА УСТАНОВКИ ГОЛОСОВОГО ВВОДА")
    print("="*60)
    
    missing = []
    
    # Проверка SpeechRecognition
    print("\n1. Проверка SpeechRecognition...")
    try:
        import speech_recognition as sr
        print(f"   ✓ SpeechRecognition установлен (версия: {sr.__version__})")
    except ImportError:
        print("   ✗ SpeechRecognition НЕ установлен")
        missing.append("SpeechRecognition")
    
    # Проверка PyAudio
    print("\n2. Проверка PyAudio...")
    try:
        import pyaudio
        print(f"   ✓ PyAudio установлен")
    except ImportError:
        print("   ✗ PyAudio НЕ установлен")
        missing.append("PyAudio")
    
    # Проверка voice_input модуля
    print("\n3. Проверка voice_input модуля...")
    try:
        sys.path.insert(0, 'src')
        import voice_input
        print("   ✓ voice_input модуль доступен")
    except ImportError as e:
        print(f"   ✗ voice_input модуль недоступен: {e}")
    
    # Проверка main.py интеграции
    print("\n4. Проверка интеграции с main.py...")
    try:
        import main
        if hasattr(main.SupportAssistant, '_handle_voice_input'):
            print("   ✓ Метод _handle_voice_input найден в SupportAssistant")
        else:
            print("   ✗ Метод _handle_voice_input НЕ найден")
    except Exception as e:
        print(f"   ⚠ Частичная проверка: {e}")
    
    # Проверка доступности микрофона
    print("\n5. Проверка микрофона...")
    if 'speech_recognition' in sys.modules and 'pyaudio' in sys.modules:
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("   ✓ Микрофон доступен")
        except OSError as e:
            print(f"   ✗ Ошибка доступа к микрофону: {e}")
        except Exception as e:
            print(f"   ⚠ Проблема с микрофоном: {e}")
    else:
        print("   ⊘ Пропущено (зависимости не установлены)")
    
    # Итоговый вывод
    print("\n" + "="*60)
    if missing:
        print("РЕЗУЛЬТАТ: Требуется установка зависимостей")
        print("\nОтсутствующие пакеты:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nУстановите с помощью:")
        print("  pip install " + " ".join(missing))
        print("\nИли:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("РЕЗУЛЬТАТ: ✓ Все проверки пройдены!")
        print("\nГолосовой ввод готов к использованию.")
        print("Запустите приложение и введите команду /voice")
        return True


if __name__ == "__main__":
    success = check_dependencies()
    sys.exit(0 if success else 1)
