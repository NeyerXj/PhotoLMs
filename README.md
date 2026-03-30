# PhotoLM

Нативное macOS приложение на SwiftUI (MVVM) для Viewer-редактирования изображений с inpainting (LaMa) и упаковкой в `.app` с вшитым Python runtime.

## Вариант без Xcode (только Python, для новичков)

Этот вариант не требует сборки `.app`. Ты запускаешь Viewer прямо через Python.

### 1) Установить Python 3.11

Скачай и установи Python с [python.org](https://www.python.org/downloads/macos/).

Проверь:

```bash
python3 --version
```

### 2) Скачать проект

```bash
git clone https://github.com/NeyerXj/PhotoLMs.git
cd PhotoLMs
```

### 3) Установить зависимости одной командой

```bash
./Scripts/python_setup.sh
```

### 4) Положить картинки

- Исходники: в папку `input/`
- Редактируемые файлы: в папку `output/` (скрипт сам скопирует из `input`)

### 5) Запустить Viewer

```bash
./Scripts/python_run_viewer.sh
```

Клавиши в Viewer:

- `ЛКМ` рисовать маску
- `ПКМ` стирать маску
- `E` применить inpainting
- `S` сохранить
- `N/P` следующее/предыдущее изображение
- `Q` выход

Дополнительно:

- запуск со своими папками:
  ```bash
  ./Scripts/python_run_viewer.sh /путь/к/input /путь/к/output
  ```
- если нужно каждый запуск полностью перезаписывать `output` из `input`:
  ```bash
  OVERWRITE=1 ./Scripts/python_run_viewer.sh
  ```

## Возможности

- Viewer-only UI: выбор `input`/`output`, запуск редактора, сохранение правок
- Ручная маска кистью + inpainting по выделенной области
- Снижение потребления RAM через ROI-обработку и ограничение размера области
- Сборка universal macOS binary (`arm64 + x86_64`)
- Вшитый `Python.framework` и зависимости внутри `.app`

## Требования

- macOS 14+
- Xcode Command Line Tools (`xcode-select --install`)
- Swift 6.2 toolchain (Xcode 16+)
- Python 3.11 (рекомендуется universal2 с python.org)

## Структура проекта

- `Sources/PhotoLM` — SwiftUI приложение (MVVM)
- `ui_viewer.py` — viewer с кистью и inpainting
- `ui_remove.py` — batch-обработка (CLI)
- `Scripts/package_app.sh` — сборка `.app` с embedded Python
- `requirements.embedded.txt` — зависимости для вшивания в `.app`

## Быстрый запуск (из исходников)

```bash
swift build -c debug
swift run PhotoLM
```

Или одной командой (пересобрать и открыть `.app`):

```bash
./Scripts/compile_and_run.sh
```

## Сборка `.app` с нуля для macOS (Apple Silicon + Intel)

### 1) Подготовить Python (universal2)

Рекомендуется Python с [python.org](https://www.python.org/downloads/macos/), чтобы сборка для `arm64` и `x86_64` работала из одной машины.

Пример пути:

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3
```

### 2) Запустить упаковку

```bash
EMBED_PYTHON_EXECUTABLE=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
FORCE_PYTHON_REBUILD=1 \
SIGNING_MODE=adhoc \
./Scripts/package_app.sh release
```

Результат:

- `PhotoLM.app` — universal `.app` (`arm64 + x86_64`)
- внутри `Contents/Resources` находятся:
  - `Python.framework`
  - `PythonSitePackages/arm64`
  - `PythonSitePackages/x86_64`
  - `PythonScripts/ui_viewer.py`, `ui_remove.py`

Запуск:

```bash
open PhotoLM.app
```

## Полезные команды

Проверка архитектур:

```bash
file PhotoLM.app/Contents/MacOS/PhotoLM
```

Проверка подписи:

```bash
codesign --verify --deep --strict --verbose=2 PhotoLM.app
```

Сборка release бинарника (без упаковки):

```bash
swift build -c release --arch arm64 --arch x86_64
```

## Примечание по запуску на других Mac

Текущая команда использует `SIGNING_MODE=adhoc`. Такая сборка подходит для локального использования и тестов, но для бесшовного запуска на любом Mac нужен `Developer ID` + notarization.
