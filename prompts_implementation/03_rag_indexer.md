# Промпт для имплементации: RAG Indexer

## Задача
Реализовать индексатор документов для RAG системы - сканирование, чтение и разбиение документов на чанки.

## Файл
`src/rag/indexer.py`

## Контекст
Этот модуль отвечает за первый этап RAG pipeline - подготовку документов. Он сканирует директорию `docs/`, читает файлы, разбивает на чанки и сохраняет индекс.

## Требования к имплементации

### Класс DocumentIndexer

#### `__init__(self, docs_dir: str, embeddings_path: str, chunk_size: int = 500, overlap: int = 50)`
```
Реализуй инициализацию:
1. Сохрани параметры:
   - self._docs_dir = docs_dir
   - self._embeddings_path = embeddings_path
   - self._chunk_size = chunk_size
   - self._overlap = overlap
2. Проверь существование директории docs_dir:
   if not os.path.exists(docs_dir):
       os.makedirs(docs_dir)
```

#### `index_all(self, embedding_generator: 'EmbeddingGenerator') -> IndexingResult`
```
Реализуй полную индексацию:
1. Найди все документы: files = self.scan_documents()
2. Создай списки для чанков и эмбедингов:
   all_chunks = []
   all_embeddings = []
   errors = []
3. Для каждого файла:
   try:
       text = self.read_document(file_path)
       for chunk in self.split_into_chunks(text, file_path):
           embedding = embedding_generator.generate(chunk.text)
           all_chunks.append(chunk)
           all_embeddings.append(embedding)
   except Exception as e:
       errors.append(f"{file_path}: {str(e)}")
4. Сохрани индекс: self.save_index(all_chunks, all_embeddings)
5. Верни IndexingResult:
   - total_files = len(files)
   - total_chunks = len(all_chunks)
   - indexed_files = [f for f in files if f not in errors]
   - errors = errors

Добавь вывод прогресса в консоль (print или logging).
```

#### `scan_documents(self) -> List[str]`
```
Реализуй сканирование директории:
1. Создай пустой список файлов
2. Используй os.walk для рекурсивного обхода self._docs_dir
3. Для каждого файла проверь расширение:
   - Получи расширение: ext = os.path.splitext(filename)[1].lower()
   - Если ext в SUPPORTED_EXTENSIONS - добавь полный путь
4. Отсортируй список по имени файла
5. Верни список путей

Пример:
for root, dirs, files in os.walk(self._docs_dir):
    for filename in files:
        ...
```

#### `read_document(self, file_path: str) -> str`
```
Реализуй чтение документа:
1. Открой файл с кодировкой utf-8:
   with open(file_path, 'r', encoding='utf-8') as f:
       content = f.read()
2. Удали лишние пробелы: content = content.strip()
3. Верни содержимое

При ошибках:
- FileNotFoundError - пробросить дальше
- UnicodeDecodeError - попробовать с encoding='latin-1', если не получится - пробросить
```

#### `split_into_chunks(self, text: str, source_file: str) -> Generator[DocumentChunk, None, None]`
```
Реализуй разбиение на чанки:
1. Установи начальную позицию: start = 0
2. Установи счетчик чанков: chunk_num = 0
3. Пока start < len(text):
   a. Вычисли конец чанка: end = min(start + self._chunk_size, len(text))
   b. Извлеки текст чанка: chunk_text = text[start:end]
   c. Сгенерируй ID: chunk_id = self._generate_chunk_id(source_file, chunk_num)
   d. Создай и yield DocumentChunk:
      DocumentChunk(
          chunk_id=chunk_id,
          text=chunk_text,
          source_file=source_file,
          position=start
      )
   e. Сдвинь позицию: start = start + self._chunk_size - self._overlap
   f. Увеличь счетчик: chunk_num += 1

Важно: последний чанк может быть меньше chunk_size - это нормально.
```

#### `save_index(self, chunks: List[DocumentChunk], embeddings: List[List[float]])`
```
Сохрани индекс в JSON:
1. Сформируй структуру данных:
   data = {
       "indexed_at": datetime.now().isoformat(),
       "config": {
           "chunk_size": self._chunk_size,
           "overlap": self._overlap
       },
       "chunks": []
   }
2. Для каждого чанка и эмбединга:
   data["chunks"].append({
       "id": chunk.chunk_id,
       "text": chunk.text,
       "embedding": embedding,
       "source": chunk.source_file,
       "position": chunk.position
   })
3. Создай директорию если не существует:
   os.makedirs(os.path.dirname(self._embeddings_path), exist_ok=True)
4. Сохрани в JSON:
   with open(self._embeddings_path, 'w', encoding='utf-8') as f:
       json.dump(data, f, ensure_ascii=False, indent=2)
```

#### `load_index(self) -> Optional[dict]`
```
Загрузи существующий индекс:
1. Проверь существование файла
2. Если не существует - верни None
3. Открой и прочитай JSON:
   with open(self._embeddings_path, 'r', encoding='utf-8') as f:
       return json.load(f)
```

#### `is_index_exists(self) -> bool`
```
return os.path.exists(self._embeddings_path)
```

#### `_generate_chunk_id(self, source_file: str, position: int) -> str`
```
Сгенерируй уникальный ID:
1. Извлеки имя файла без пути: filename = os.path.basename(source_file)
2. Удали расширение: name = os.path.splitext(filename)[0]
3. Сформируй ID: return f"{name}_chunk_{position:04d}"

Пример: "example_chunk_0001"
```

## Алгоритм разбиения на чанки (визуализация)

```
Текст: "ABCDEFGHIJKLMNOPQRSTUVWXYZ" (26 символов)
chunk_size = 10, overlap = 3

Чанк 1: "ABCDEFGHIJ" (позиция 0-9)
         ^------^
Чанк 2: "HIJKLMNOPQ" (позиция 7-16)
              ^------^
Чанк 3: "OPQRSTUVWX" (позиция 14-23)
                   ^------^
Чанк 4: "VWXYZ" (позиция 21-25, последний чанк короче)
                        ^--^
```

## Зависимости
- os
- json
- datetime
- typing

## Тестирование
После реализации проверь:
1. Сканирование находит только .txt и .md файлы
2. Чанки правильно перекрываются
3. JSON сохраняется корректно
4. Загрузка индекса работает
