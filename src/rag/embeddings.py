"""
Модуль генерации эмбедингов.

Отвечает за:
- Взаимодействие с локальной LLM для создания эмбедингов
- Преобразование текста в векторные представления
"""

import time
from typing import List, Optional
from dataclasses import dataclass

import requests


@dataclass
class EmbeddingConfig:
    """Конфигурация для генератора эмбедингов."""
    host: str
    port: int
    model_name: str
    endpoint: str
    timeout: int = 30
    retry_attempts: int = 3


class EmbeddingGenerator:
    """
    Генератор эмбедингов через локальную LLM.
    
    Обеспечивает:
    - Подключение к локальной LLM (например, Ollama)
    - Генерацию эмбедингов для текстовых чанков
    - Батчевую обработку для эффективности
    """
    
    def __init__(self, config: EmbeddingConfig) -> None:
        """
        Инициализация генератора.
        
        Args:
            config: Конфигурация подключения к LLM
        """
        self._config = config
        self._base_url = f"http://{config.host}:{config.port}{config.endpoint}"
        self._embedding_dim: Optional[int] = None
    
    def generate(self, text: str) -> List[float]:
        """
        Генерация эмбединга для одного текста.
        
        Args:
            text: Текст для преобразования
            
        Returns:
            Вектор эмбединга (список float)
            
        Raises:
            EmbeddingError: При ошибке генерации
        """
        response = self._retry_with_backoff(self._send_request, text)
        return self._parse_embedding(response)
    
    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Батчевая генерация эмбедингов.
        
        Args:
            texts: Список текстов
            
        Returns:
            Список эмбедингов (в том же порядке)
        
        Note:
            Ollama не поддерживает батчевые запросы напрямую,
            поэтому обрабатываем по одному.
        """
        embeddings: List[List[float]] = []
        for i, text in enumerate(texts):
            print(f"Генерация эмбединга {i+1}/{len(texts)}...")
            embedding = self.generate(text)
            embeddings.append(embedding)
        return embeddings
    
    def check_model_availability(self) -> bool:
        """
        Проверка доступности модели.
        
        Returns:
            True если модель доступна
        """
        try:
            self.generate("test")
            return True
        except Exception:
            return False
    
    def get_embedding_dimension(self) -> int:
        """
        Получение размерности эмбедингов.
        
        Returns:
            Размерность вектора эмбединга
        """
        if self._embedding_dim is not None:
            return self._embedding_dim
        
        self._embedding_dim = len(self.generate("test"))
        return self._embedding_dim
    
    def _send_request(self, text: str) -> dict:
        """
        Отправка запроса к API локальной LLM.
        
        Args:
            text: Текст для эмбединга
            
        Returns:
            JSON ответ API
            
        Raises:
            EmbeddingConnectionError: При проблемах с подключением
        """
        payload = {
            "model": self._config.model_name,
            "prompt": text
        }
        
        try:
            response = requests.post(
                self._base_url,
                json=payload,
                timeout=self._config.timeout
            )
        except requests.exceptions.ConnectionError:
            raise EmbeddingConnectionError("Не удалось подключиться к LLM")
        except requests.exceptions.Timeout:
            raise EmbeddingConnectionError("Таймаут подключения к LLM")
        
        if response.status_code != 200:
            raise EmbeddingConnectionError(f"Ошибка API: {response.status_code}")
        
        return response.json()
    
    def _parse_embedding(self, response: dict) -> List[float]:
        """
        Извлечение эмбединга из ответа API.
        
        Args:
            response: JSON ответ от API
            
        Returns:
            Вектор эмбединга
            
        Raises:
            EmbeddingParseError: При неожиданном формате ответа
        """
        if "embedding" not in response:
            raise EmbeddingParseError("Отсутствует поле 'embedding' в ответе")
        return response["embedding"]
    
    def _retry_with_backoff(self, func: callable, *args, **kwargs):
        """
        Выполнение функции с retry и экспоненциальным backoff.
        
        Args:
            func: Функция для выполнения
            *args, **kwargs: Аргументы функции
            
        Returns:
            Результат функции
            
        Raises:
            EmbeddingConnectionError: После исчерпания попыток
        """
        max_attempts = self._config.retry_attempts
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except EmbeddingConnectionError as e:
                if attempt == max_attempts - 1:
                    raise
                wait_time = 2 ** attempt  # 1, 2, 4 секунды
                print(f"Попытка {attempt + 1} не удалась. Ожидание {wait_time}с...")
                time.sleep(wait_time)


class EmbeddingError(Exception):
    """Базовый класс ошибок генерации эмбедингов."""
    pass


class EmbeddingConnectionError(EmbeddingError):
    """Ошибка подключения к LLM."""
    pass


class EmbeddingParseError(EmbeddingError):
    """Ошибка парсинга ответа."""
    pass
