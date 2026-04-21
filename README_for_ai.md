# README for AI

## Как работает программа

Программа обрабатывает изображения из входной папки, удаляет текст (режим `text`) и сохраняет результат в выходную папку.  
После обработки автоматически открывается Viewer для ручной доработки маской и сохранения.

Поток работы:

1. Берёт изображения из `input/`
2. Запускает удаление текста через `ui_remove.py`
3. Сохраняет обработанные файлы в `output/`
4. Открывает Viewer (`--open-viewer`) для проверки и правок

## Основная команда (рекомендуемая)

```bash
python ui_remove.py \
  --input input \
  --output output \
  --mode text \
  --device auto \
  --open-viewer
```

## Команда для старого Mac / малого объёма ОЗУ

```bash
python ui_remove.py \
  --input input \
  --output output \
  --mode text \
  --device cpu \
  --lama-device cpu \
  --max-side 1280 \
  --open-viewer
```

## Какие папки использовать

- `input/`:
  - сюда класть исходные изображения для обработки
  - можно использовать вложенные папки
- `output/`:
  - сюда программа записывает обработанные изображения
  - из этой папки Viewer открывает результаты

Если папок нет, создай:

```bash
mkdir -p input output
```

## Быстрый запуск

```bash
python ui_remove.py --input input --output output --mode text --device auto --open-viewer
```
