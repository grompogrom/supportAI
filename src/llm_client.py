"""
Клиент для работы с Perplexity API.

Обеспечивает взаимодействие с LLM моделью sonar-pro
через Perplexity API.
"""

import json
import requests
from typing import List, Dict, Any


class LLMClient:
    """
    Клиент для Perplexity API.
    
    Обеспечивает:
    - Отправку запросов к модели sonar-pro
    - Управление историей сообщений
    - Обработку ответов с tool calls
    """
    
    API_BASE_URL = "https://api.perplexity.ai"
    MODEL_NAME = "sonar-pro"
    
    def __init__(self, api_key: str, system_prompt: str) -> None:
        """
        Инициализация клиента.
        
        Args:
            api_key: API ключ Perplexity
            system_prompt: Системный промпт для модели
        """
        self._api_key = api_key
        self._system_prompt = system_prompt
        self._messages: List[Dict[str, str]] = []
        self._messages.append({"role": "system", "content": system_prompt})
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def send_message(self, message: str) -> str:
        """
        Отправка сообщения в LLM.
        
        Args:
            message: Текст сообщения пользователя
            
        Returns:
            Текст ответа от модели
        """
        self._messages.append({"role": "user", "content": message})
        
        payload = self._build_request_payload()
        
        response = requests.post(
            f"{self.API_BASE_URL}/chat/completions",
            headers=self._headers,
            json=payload
        )
        
        if response.status_code != 200:
            self._handle_api_error(response.status_code, response.text)
        
        response_text = self._parse_response(response.json())
        
        self._messages.append({"role": "assistant", "content": response_text})
        
        return response_text
    
    def send_tool_result(self, tool_name: str, result: Any) -> str:
        """
        Отправка результата выполнения инструмента в LLM.
        
        Args:
            tool_name: Имя вызванного инструмента
            result: Результат выполнения инструмента
            
        Returns:
            Новый ответ модели с учетом результата
        """
        message = f"Результат выполнения инструмента {tool_name}:\n{json.dumps(result, ensure_ascii=False)}"
        
        self._messages.append({"role": "user", "content": message})
        
        payload = self._build_request_payload()
        
        response = requests.post(
            f"{self.API_BASE_URL}/chat/completions",
            headers=self._headers,
            json=payload
        )
        
        if response.status_code != 200:
            self._handle_api_error(response.status_code, response.text)
        
        response_text = self._parse_response(response.json())
        
        self._messages.append({"role": "assistant", "content": response_text})
        
        return response_text
    
    def get_messages_history(self) -> List[Dict[str, str]]:
        """
        Получение истории сообщений.
        
        Returns:
            Копия списка сообщений в формате [{role, content}, ...]
        """
        return self._messages.copy()
    
    def clear_history(self) -> None:
        """
        Очистка истории сообщений.
        Системный промпт сохраняется.
        """
        self._messages = [{"role": "system", "content": self._system_prompt}]
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Установка нового системного промпта.
        
        Args:
            prompt: Новый системный промпт
        """
        self._system_prompt = prompt
        self._messages[0] = {"role": "system", "content": prompt}
    
    def _build_request_payload(self) -> Dict[str, Any]:
        """
        Формирование payload для API запроса.
        
        Returns:
            Словарь с параметрами запроса
        """
        return {
            "model": self.MODEL_NAME,
            "messages": self._messages,
            "temperature": 0.7,
            "max_tokens": 2048,
            "disable_search": True
        }
    
    def _parse_response(self, response_json: Dict[str, Any]) -> str:
        """
        Парсинг ответа API.
        
        Args:
            response_json: JSON ответ от API
            
        Returns:
            Текст ответа модели
            
        Raises:
            ValueError: Если ответ имеет неожиданный формат
        """
        if "choices" not in response_json:
            raise ValueError("Ответ API не содержит ключ 'choices'")
        
        try:
            return response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Неожиданный формат ответа API: {e}")
    
    def _handle_api_error(self, status_code: int, response_text: str) -> None:
        """
        Обработка ошибок API.
        
        Args:
            status_code: HTTP код ответа
            response_text: Тело ответа с ошибкой
            
        Raises:
            AuthenticationError: При проблемах с API ключом
            RateLimitError: При превышении лимитов
            APIError: При других ошибках API
        """
        if status_code == 401:
            raise AuthenticationError("Неверный API ключ")
        elif status_code == 429:
            raise RateLimitError("Превышен лимит запросов")
        else:
            raise APIError(f"Ошибка API: {status_code} - {response_text}")


class LLMError(Exception):
    """Базовый класс ошибок LLM клиента."""
    pass


class AuthenticationError(LLMError):
    """Ошибка аутентификации (неверный API ключ)."""
    pass


class RateLimitError(LLMError):
    """Ошибка превышения лимита запросов."""
    pass


class APIError(LLMError):
    """Общая ошибка API."""
    pass
