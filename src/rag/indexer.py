"""
Модуль индексации документов.

Отвечает за:
- Сканирование директории docs/
- Чтение файлов различных форматов
- Разбиение на чанки
- Координацию процесса индексации
"""

import os
import json
from datetime import datetime
from typing import List, Generator, Optional
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """Чанк документа."""
    chunk_id: str
    text: str
    source_file: str
    position: int  # Позиция чанка в исходном файле
    

@dataclass
class IndexingResult:
    """Результат индексации."""
    total_files: int
    total_chunks: int
    indexed_files: List[str]
    errors: List[str]


class DocumentIndexer:
    """
    Индексатор документов для RAG системы.
    
    Обеспечивает:
    - Сканирование директории с документами
    - Чтение файлов .txt, .md
    - Разбиение текста на чанки с перекрытием
    - Сохранение индекса в JSON
    """
    
    SUPPORTED_EXTENSIONS = [".txt", ".md"]
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_OVERLAP = 50
    
    def __init__(self, docs_dir: str, embeddings_path: str,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 overlap: int = DEFAULT_OVERLAP) -> None:
        """
        Инициализация индексатора.
        
        Args:
            docs_dir: Путь к директории с документами
            embeddings_path: Путь к файлу для сохранения эмбедингов
            chunk_size: Размер чанка в символах (по умолчанию 500)
            overlap: Размер перекрытия между чанками (по умолчанию 50)
            
        Действия:
        - Сохранить пути и параметры
        - Проверить существование директории docs
        """
        self._docs_dir = docs_dir
        self._embeddings_path = embeddings_path
        self._chunk_size = chunk_size
        self._overlap = overlap
        
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir)
    
    def index_all(self, embedding_generator: 'EmbeddingGenerator') -> IndexingResult:
        """
        Индексация всех документов в директории.
        
        Args:
            embedding_generator: Генератор эмбедингов
            
        Returns:
            Результат индексации со статистикой
            
        Действия:
        - Найти все файлы с поддерживаемыми расширениями
        - Для каждого файла:
          - Прочитать содержимое
          - Разбить на чанки
          - Сгенерировать эмбединги
        - Сохранить все в JSON файл
        - Вернуть статистику
        """
        files = self.scan_documents()
        all_chunks: List[DocumentChunk] = []
        all_embeddings: List[List[float]] = []
        errors: List[str] = []
        error_files: set = set()
        
        print(f"Найдено файлов для индексации: {len(files)}")
        
        for file_idx, file_path in enumerate(files):
            print(f"Обработка файла {file_idx + 1}/{len(files)}: {file_path}")
            try:
                text = self.read_document(file_path)
                chunk_count = 0
                for chunk in self.split_into_chunks(text, file_path):
                    print(f"  Генерация эмбединга для чанка {chunk.chunk_id}...")
                    embedding = embedding_generator.generate(chunk.text)
                    all_chunks.append(chunk)
                    all_embeddings.append(embedding)
                    chunk_count += 1
                print(f"  Создано чанков: {chunk_count}")
            except Exception as e:
                error_msg = f"{file_path}: {str(e)}"
                errors.append(error_msg)
                error_files.add(file_path)
                print(f"  Ошибка: {str(e)}")
        
        self.save_index(all_chunks, all_embeddings)
        print(f"Индекс сохранён: {len(all_chunks)} чанков")
        
        indexed_files = [f for f in files if f not in error_files]
        
        return IndexingResult(
            total_files=len(files),
            total_chunks=len(all_chunks),
            indexed_files=indexed_files,
            errors=errors
        )
    
    def scan_documents(self) -> List[str]:
        """
        Сканирование директории на наличие документов.
        
        Returns:
            Список путей к найденным файлам
            
        Действия:
        - Рекурсивно обойти директорию docs/
        - Отфильтровать файлы по расширению
        - Вернуть список путей
        """
        found_files: List[str] = []
        
        for root, dirs, files in os.walk(self._docs_dir):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    full_path = os.path.join(root, filename)
                    found_files.append(full_path)
        
        found_files.sort()
        return found_files
    
    def read_document(self, file_path: str) -> str:
        """
        Чтение содержимого документа.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Текстовое содержимое файла
            
        Raises:
            FileNotFoundError: Если файл не найден
            UnicodeDecodeError: При ошибке кодировки
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        return content.strip()
    
    def split_into_chunks(self, text: str, source_file: str) -> Generator[DocumentChunk, None, None]:
        """
        Разбиение текста на чанки.
        
        Args:
            text: Исходный текст документа
            source_file: Путь к исходному файлу
            
        Yields:
            DocumentChunk объекты
            
        Алгоритм:
        - Начать с позиции 0
        - Взять chunk_size символов
        - Сдвинуться на (chunk_size - overlap) символов
        - Повторять до конца текста
        - Генерировать уникальный chunk_id для каждого чанка
        """
        start = 0
        chunk_num = 0
        
        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            chunk_text = text[start:end]
            chunk_id = self._generate_chunk_id(source_file, chunk_num)
            
            yield DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source_file=source_file,
                position=start
            )
            
            start = start + self._chunk_size - self._overlap
            chunk_num += 1
    
    def save_index(self, chunks: List[DocumentChunk], 
                   embeddings: List[List[float]]) -> None:
        """
        Сохранение индекса в JSON файл.
        
        Args:
            chunks: Список чанков
            embeddings: Соответствующие эмбединги
            
        Формат JSON:
        {
            "indexed_at": "ISO timestamp",
            "config": {
                "chunk_size": 500,
                "overlap": 50
            },
            "chunks": [
                {
                    "id": "chunk_001",
                    "text": "...",
                    "embedding": [...],
                    "source": "docs/file.md",
                    "position": 0
                }
            ]
        }
        """
        data = {
            "indexed_at": datetime.now().isoformat(),
            "config": {
                "chunk_size": self._chunk_size,
                "overlap": self._overlap
            },
            "chunks": []
        }
        
        for chunk, embedding in zip(chunks, embeddings):
            data["chunks"].append({
                "id": chunk.chunk_id,
                "text": chunk.text,
                "embedding": embedding,
                "source": chunk.source_file,
                "position": chunk.position
            })
        
        # Создаём директорию если не существует
        dir_path = os.path.dirname(self._embeddings_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(self._embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_index(self) -> Optional[dict]:
        """
        Загрузка существующего индекса.
        
        Returns:
            Словарь с индексом или None если файл не существует
        """
        if not os.path.exists(self._embeddings_path):
            return None
        
        with open(self._embeddings_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def is_index_exists(self) -> bool:
        """
        Проверка наличия файла индекса.
        
        Returns:
            True если индекс существует
        """
        return os.path.exists(self._embeddings_path)
    
    def _generate_chunk_id(self, source_file: str, position: int) -> str:
        """
        Генерация уникального ID для чанка.
        
        Args:
            source_file: Путь к исходному файлу
            position: Номер чанка (позиция в последовательности)
            
        Returns:
            Уникальный идентификатор чанка
        """
        filename = os.path.basename(source_file)
        name = os.path.splitext(filename)[0]
        return f"{name}_chunk_{position:04d}"


class IndexerError(Exception):
    """Базовый класс ошибок индексатора."""
    pass
