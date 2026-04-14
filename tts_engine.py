"""
SignQazaq Pro — Асинхронный TTS-движок (Text-to-Speech Engine)

Использует встроенную команду macOS `say` через subprocess.
Озвучка запускается в отдельном потоке — не блокирует видеопоток.

Доступные голоса:
  - Aru    (kk_KZ) — Казахский 🇰🇿  ← приоритетный
  - Milena (ru_RU) — Русский   🇷🇺  ← запасной
"""

import subprocess
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class TTSRequest:
    text:  str
    voice: str
    rate:  int  # Скорость речи (слов в минуту, default ~175)


class TTSEngine:
    """
    Асинхронный TTS-движок с очередью и приоритетом казахского голоса.

    Использование:
        tts = TTSEngine()
        tts.speak("Сәлем")           # Казахский
        tts.speak("Привет", lang='ru')  # Русский
        tts.stop()                   # При завершении программы
    """

    # Приоритетный список голосов
    VOICE_KK = "Aru"        # Казахский (kk_KZ)
    VOICE_RU = "Milena"     # Русский   (ru_RU)

    def __init__(self, rate: int = 160):
        self.rate = rate
        self._queue:   queue.Queue[Optional[TTSRequest]] = queue.Queue()
        self._lock     = threading.Lock()
        self._speaking = False
        self._active_voice: Optional[str] = None

        # Определяем доступные голоса
        self._available_voices = self._detect_voices()
        self._default_voice    = self._pick_default_voice()

        # Запускаем фоновый поток-воркер
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,   # Поток умирает вместе с основным процессом
            name="TTS-Worker",
        )
        self._worker_thread.start()

        print(f"  🔊 TTS готов | Голос: {self._default_voice} | Скорость: {self.rate}")

    # ──────────────────────────────────────────
    # Публичный API
    # ──────────────────────────────────────────

    def speak(self, text: str, lang: str = "kk") -> None:
        """
        Добавить текст в очередь озвучки.

        Args:
            text: Текст для произношения
            lang: 'kk' (казахский, default) или 'ru' (русский)
        """
        if not text or not text.strip():
            return

        voice = self._pick_voice_for_lang(lang)

        # Очищаем очередь от устаревших запросов (актуален только последний)
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        self._queue.put(TTSRequest(text=text.strip(), voice=voice, rate=self.rate))

    def speak_letter(self, letter: str) -> None:
        """
        Озвучить одну букву (с небольшим ускорением темпа).
        """
        self.speak(letter, lang="kk")

    def speak_phrase(self, text: str) -> None:
        """
        Озвучить целую фразу/предложение медленнее для чёткости.
        """
        if not text or not text.strip():
            return
        voice = self._pick_voice_for_lang("kk")
        self._queue.put(TTSRequest(text=text.strip(), voice=voice, rate=130))

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    @property
    def default_voice(self) -> str:
        return self._default_voice

    def stop(self) -> None:
        """Остановить движок (вызвать при завершении программы)."""
        self._queue.put(None)  # Сигнал остановки воркеру

    # ──────────────────────────────────────────
    # Внутренние методы
    # ──────────────────────────────────────────

    def _worker_loop(self) -> None:
        """Фоновый поток: ждёт задачи из очереди и озвучивает."""
        while True:
            request = self._queue.get()

            # None = сигнал завершения
            if request is None:
                break

            with self._lock:
                self._speaking = True
                self._active_voice = request.voice

            try:
                self._say(request.text, request.voice, request.rate)
            except Exception as e:
                print(f"  ⚠️  TTS ошибка: {e}")
            finally:
                with self._lock:
                    self._speaking = False
                    self._active_voice = None

    def _say(self, text: str, voice: str, rate: int) -> None:
        """Запускает команду `say` синхронно (внутри воркер-потока)."""
        cmd = ["say", "-v", voice, "-r", str(rate), text]
        subprocess.run(cmd, check=False, capture_output=True)

    def _detect_voices(self) -> set[str]:
        """Получает список доступных голосов `say`."""
        try:
            result = subprocess.run(
                ["say", "-v", "?"],
                capture_output=True, text=True, timeout=5
            )
            lines  = result.stdout.splitlines()
            voices = {line.split()[0] for line in lines if line.strip()}
            return voices
        except Exception:
            return set()

    def _pick_default_voice(self) -> str:
        """Выбирает лучший доступный голос."""
        if self.VOICE_KK in self._available_voices:
            return self.VOICE_KK
        if self.VOICE_RU in self._available_voices:
            return self.VOICE_RU
        return "Alex"  # Запасной English-голос

    def _pick_voice_for_lang(self, lang: str) -> str:
        if lang == "ru" and self.VOICE_RU in self._available_voices:
            return self.VOICE_RU
        if lang == "kk" and self.VOICE_KK in self._available_voices:
            return self.VOICE_KK
        return self._default_voice
