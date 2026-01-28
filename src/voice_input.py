"""
Модуль для обработки голосового ввода.

Предоставляет функциональность распознавания речи с возможностью
прерывания по нажатию Enter.
"""

import threading
import sys
from typing import Optional
import speech_recognition as sr


class VoiceInputHandler:
    """
    Обработчик голосового ввода.
    
    Использует библиотеку SpeechRecognition для распознавания речи
    с микрофона и позволяет прервать запись нажатием Enter.
    """
    
    def __init__(self, language: str = "ru-RU"):
        """
        Инициализация обработчика голосового ввода.
        
        Args:
            language: Язык распознавания (по умолчанию "ru-RU" - русский)
        """
        self.language = language
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        self.audio_data = None
        self.stop_listening = None
        self.is_stopped = threading.Event()
    
    def listen_until_enter(self) -> Optional[str]:
        """
        Начать прослушивание с микрофона до нажатия Enter.
        
        Returns:
            Распознанный текст или None при ошибке/отмене
            
        Raises:
            OSError: Если микрофон недоступен
            Exception: При других ошибках распознавания
        """
        try:
            # Проверка доступности микрофона и калибровка
            microphone = sr.Microphone()
            
            # Калибровка в отдельном контексте
            print("🎤 Калибровка микрофона... (пожалуйста, подождите)")
            with microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            print("✓ Готово! Говорите... (нажмите Enter для остановки)")
            
            # Запуск фонового прослушивания
            audio_queue = []
            
            def callback(recognizer, audio):
                """Коллбэк для накопления аудио данных."""
                audio_queue.append(audio)
            
            # Создаем новый экземпляр микрофона для listen_in_background
            # (listen_in_background управляет контекстом самостоятельно)
            source_for_listening = sr.Microphone()
            
            # Начинаем прослушивание в фоне
            self.stop_listening = self.recognizer.listen_in_background(
                source_for_listening, 
                callback,
                phrase_time_limit=30  # Максимум 30 секунд на одну фразу
            )
            
            # Ожидание нажатия Enter
            try:
                input()  # Блокирующий вызов до нажатия Enter
            except KeyboardInterrupt:
                pass
            
            # Остановка прослушивания
            if self.stop_listening:
                self.stop_listening(wait_for_stop=False)
            
            # Обработка накопленных аудио данных
            if not audio_queue:
                print("\n⚠️  Аудио не обнаружено")
                return None
            
            # Объединяем все аудио сегменты
            print("\n🔄 Обработка записи...")
            all_text = []
            
            for i, audio in enumerate(audio_queue):
                try:
                    text = self.recognizer.recognize_google(
                        audio,
                        language=self.language
                    )
                    if text:
                        all_text.append(text)
                except sr.UnknownValueError:
                    # Фрагмент не распознан - пропускаем
                    continue
                except sr.RequestError as e:
                    print(f"\n❌ Ошибка сервиса распознавания: {e}")
                    print("💡 Проверьте подключение к интернету")
                    return None
            
            if not all_text:
                print("\n⚠️  Не удалось распознать речь")
                return None
            
            # Объединяем все распознанные фрагменты
            result = " ".join(all_text)
            return result
                
        except OSError as e:
            print(f"\n❌ Ошибка доступа к микрофону: {e}")
            print("💡 Убедитесь, что микрофон подключен и разрешен доступ")
            return None
        except Exception as e:
            print(f"\n❌ Неожиданная ошибка: {e}")
            return None
    
    def __del__(self):
        """Очистка ресурсов при удалении объекта."""
        if self.stop_listening:
            try:
                self.stop_listening(wait_for_stop=False)
            except:
                pass


def test_voice_input():
    """
    Тестовая функция для проверки голосового ввода.
    
    Может быть запущена напрямую:
        python voice_input.py
    """
    print("=== Тест голосового ввода ===\n")
    handler = VoiceInputHandler(language="ru-RU")
    
    try:
        result = handler.listen_until_enter()
        if result:
            print(f"\n✓ Распознанный текст: {result}")
        else:
            print("\n✗ Распознавание не удалось")
    except KeyboardInterrupt:
        print("\n\nТест прерван пользователем")


if __name__ == "__main__":
    test_voice_input()
