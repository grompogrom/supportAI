"""
Модуль поиска релевантных документов.

Отвечает за:
- Загрузку индекса эмбедингов
- Поиск по косинусному сходству
- Возврат топ-K релевантных чанков
"""

import json
import os
from typing import List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass

import numpy as np

if TYPE_CHECKING:
    from .embeddings import EmbeddingGenerator


@dataclass
class SearchResult:
    """Результат поиска."""
    chunk_id: str
    text: str
    source_file: str
    similarity_score: float
    position: int


class DocumentRetriever:
    """
    Поиск релевантных документов в RAG системе.
    
    Обеспечивает:
    - Загрузку индекса эмбедингов
    - Поиск по косинусному сходству
    - Возврат топ-K наиболее релевантных чанков
    """
    
    DEFAULT_TOP_K = 3
    
    def __init__(self, embeddings_path: str, 
                 embedding_generator: 'EmbeddingGenerator') -> None:
        """
        Инициализация retriever'а.
        
        Args:
            embeddings_path: Путь к файлу с индексом
            embedding_generator: Генератор эмбедингов для запросов
            
        Действия:
        - Сохранить пути
        - Сохранить ссылку на генератор эмбедингов
        - Загрузить индекс в память (ленивая загрузка)
        """
        self._embeddings_path = embeddings_path
        self._embedding_generator = embedding_generator
        self._index: Optional[dict] = None
        self._index_mtime: Optional[float] = None
    
    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[SearchResult]:
        """
        Поиск релевантных чанков по запросу.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов (по умолчанию 3)
            
        Returns:
            Список SearchResult, отсортированный по релевантности
            
        Действия:
        - Преобразовать запрос в эмбединг
        - Вычислить косинусное сходство со всеми чанками
        - Отсортировать по убыванию сходства
        - Вернуть топ-K результатов
        """
        # Проверь и загрузи индекс
        self._reload_index_if_needed()
        if self._index is None:
            raise IndexNotFoundError("Индекс не найден. Запустите /index")
        
        # Сгенерируй эмбединг запроса
        query_embedding = self._embedding_generator.generate(query)
        
        # Вычисли сходство со всеми чанками
        similarities = self._compute_all_similarities(query_embedding)
        
        # Отсортируй по убыванию сходства
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Возьми топ-K результатов
        top_results = similarities[:top_k]
        
        # Сформируй SearchResult для каждого
        results = []
        for idx, score in top_results:
            chunk = self._index["chunks"][idx]
            results.append(SearchResult(
                chunk_id=chunk["id"],
                text=chunk["text"],
                source_file=chunk["source"],
                similarity_score=score,
                position=chunk["position"]
            ))
        return results
    
    def load_index(self) -> bool:
        """
        Загрузка индекса в память.
        
        Returns:
            True если индекс успешно загружен
            
        Raises:
            RetrieverError: Если индекс не найден или поврежден
        """
        # Проверь существование файла
        if not os.path.exists(self._embeddings_path):
            return False
        
        # Загрузи JSON
        with open(self._embeddings_path, 'r', encoding='utf-8') as f:
            self._index = json.load(f)
        
        # Проверь наличие чанков
        if not self._index.get("chunks"):
            raise IndexCorruptedError("Индекс пуст или поврежден")
        
        # Сохрани время модификации
        self._index_mtime = os.path.getmtime(self._embeddings_path)
        return True
    
    def is_index_loaded(self) -> bool:
        """
        Проверка загружен ли индекс.
        
        Returns:
            True если индекс в памяти
        """
        return self._index is not None
    
    def get_index_stats(self) -> dict:
        """
        Получение статистики по индексу.
        
        Returns:
            Словарь со статистикой:
            - total_chunks: количество чанков
            - indexed_at: дата индексации
            - source_files: список исходных файлов
        """
        # Если индекс не загружен - загрузи
        if self._index is None:
            self.load_index()
        
        if self._index is None:
            raise IndexNotFoundError("Индекс не найден")
        
        # Собери статистику
        return {
            "total_chunks": len(self._index["chunks"]),
            "indexed_at": self._index.get("indexed_at"),
            "source_files": list(set(c["source"] for c in self._index["chunks"])),
            "config": self._index.get("config", {})
        }
    
    def format_results_for_llm(self, results: List[SearchResult]) -> str:
        """
        Форматирование результатов для отправки в LLM.
        
        Args:
            results: Список результатов поиска
            
        Returns:
            Форматированная строка с контекстом
            
        Формат:
        Найденные документы:
        
        [1] Источник: docs/file.md (релевантность: 0.95)
        <текст чанка>
        
        [2] ...
        """
        output = "Найденные документы:\n\n"
        for i, result in enumerate(results, 1):
            output += f"[{i}] Источник: {result.source_file} "
            output += f"(релевантность: {result.similarity_score:.2f})\n"
            output += f"{result.text}\n\n"
        return output.strip()
    
    def _compute_cosine_similarity(self, vec1: List[float], 
                                   vec2: List[float]) -> float:
        """
        Вычисление косинусного сходства между векторами.
        
        Args:
            vec1: Первый вектор
            vec2: Второй вектор
            
        Returns:
            Значение косинусного сходства от -1 до 1
            
        Формула:
        cos(θ) = (A · B) / (||A|| * ||B||)
        """
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def _compute_all_similarities(self, query_embedding: List[float]) -> List[Tuple[int, float]]:
        """
        Вычисление сходства запроса со всеми чанками.
        
        Args:
            query_embedding: Эмбединг запроса
            
        Returns:
            Список кортежей (индекс_чанка, сходство)
        """
        similarities = []
        for idx, chunk in enumerate(self._index["chunks"]):
            similarity = self._compute_cosine_similarity(
                query_embedding,
                chunk["embedding"]
            )
            similarities.append((idx, similarity))
        return similarities
    
    def _reload_index_if_needed(self) -> None:
        """
        Перезагрузка индекса если файл изменился.
        
        Действия:
        - Проверить время модификации файла
        - Перезагрузить если файл обновлен
        """
        # Если индекс не загружен - загрузи
        if self._index is None:
            self.load_index()
            return
        
        # Проверь время модификации файла
        if os.path.exists(self._embeddings_path):
            current_mtime = os.path.getmtime(self._embeddings_path)
            if current_mtime != self._index_mtime:
                print("Индекс обновлен, перезагружаем...")
                self.load_index()


class RetrieverError(Exception):
    """Базовый класс ошибок retriever'а."""
    pass


class IndexNotFoundError(RetrieverError):
    """Индекс не найден."""
    pass


class IndexCorruptedError(RetrieverError):
    """Индекс поврежден."""
    pass
