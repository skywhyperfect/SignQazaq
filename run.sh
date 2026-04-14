#!/bin/bash
# ─────────────────────────────────────────────
# SignQazaq Pro — скрипт запуска
# Использование: ./run.sh
# ─────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Виртуальное окружение не найдено."
    echo "   Создай его командой:"
    echo "   /opt/homebrew/bin/python3.11 -m venv venv"
    echo "   venv/bin/pip install -r requirements.txt"
    exit 1
fi

echo "🚀 Запуск SignQazaq Pro v0.4 (жесты + транскрипция + TTS)..."
cd "$SCRIPT_DIR"
"$VENV_PYTHON" main.py
