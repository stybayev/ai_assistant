# AI Assistant

Локальный RAG-ассистент по документации [Ollama](https://ollama.com).
Всё работает офлайн — LLM крутится в Ollama, эмбеддинги считаются через `sentence-transformers`, поиск — через FAISS.

## Структура

| Файл / папка         | Назначение                                                        |
|----------------------|-------------------------------------------------------------------|
| `check_env.ipynb`    | Проверка окружения: версии Python, numpy, sentence-transformers, faiss |
| `check_ollama.ipynb` | Первый HTTP-запрос к локальному Ollama (`GET /api/tags`)          |
| `chunking.ipynb`     | Загрузка `docs/` и нарезка Markdown на чанки для RAG              |
| `docs/`              | Документация Ollama (`.md`/`.mdx`) — источник данных для ассистента |
| `requirements.txt`   | Зависимости Python                                                |

## Запуск

### 1. Установить Ollama и скачать модель

```bash
# macOS
brew install ollama
ollama serve            # запускает сервер на http://localhost:11434
ollama pull llama3.2    # любая модель на ваш выбор
```

Инструкции для Linux/Windows — в `docs/linux.mdx` и `docs/windows.mdx`.

### 2. Создать виртуальное окружение и установить зависимости

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Запустить Jupyter

```bash
jupyter lab
```

### 4. Пройти ноутбуки по порядку

1. `check_env.ipynb` — убедиться, что все библиотеки импортируются.
2. `check_ollama.ipynb` — убедиться, что Ollama отвечает и модель скачана.
3. `chunking.ipynb` — нарезать `docs/` на чанки (первый шаг RAG-пайплайна).

## Что дальше

Следующие шаги пайплайна — эмбеддинги чанков, индекс FAISS, retrieval и генерация ответа через Ollama — будут добавлены отдельными ноутбуками.
