"""
SignQazaq Pro — Геометрический классификатор жестов (Gesture Engine)

Алгоритм:
  1. Принимает 21 нормализованных landmark MediaPipe (x, y, z)
  2. Определяет состояние каждого пальца (разогнут / согнут)
  3. Маппит комбинацию пальцев на букву казахского жестового языка
  4. Дебаунсинг: жест подтверждается только после N стабильных кадров

Примечание: маппинг приближённый (геометрический), не настоящий КЖЯ.
После сбора датасета — заменить classify() на ML-модель (train.py).
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# Индексы ключевых точек MediaPipe Hands
# ─────────────────────────────────────────────
WRIST        = 0
THUMB_CMC    = 1;  THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP   = 4
INDEX_MCP    = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP   = 8
MIDDLE_MCP   = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP  = 12
RING_MCP     = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP    = 16
PINKY_MCP    = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP   = 20


# ─────────────────────────────────────────────
# Таблица жестов: (большой, указ, средн, безым, мизинец) → (буква, описание kz)
# ─────────────────────────────────────────────
GESTURE_MAP: dict[tuple, tuple[str, str]] = {
    # Все пальцы
    (1, 1, 1, 1, 1): ("А",  "Ашық алақан"),       # открытая ладонь
    (0, 0, 0, 0, 0): ("Ж",  "Жұдырық"),           # кулак
    # Один палец
    (0, 1, 0, 0, 0): ("Б",  "Сілтеу саусағы"),    # указательный
    (1, 0, 0, 0, 0): ("Д",  "Бас бармақ"),        # большой (thumbs up)
    (0, 0, 0, 0, 1): ("З",  "Кіші саусақ"),       # мизинец
    (0, 0, 1, 0, 0): ("Қ",  "Орта саусақ"),       # средний
    # Два пальца
    (0, 1, 1, 0, 0): ("В",  "Виктория"),          # указательный + средний
    (1, 1, 0, 0, 0): ("Е",  "Мылтық"),            # большой + указательный
    (0, 1, 0, 0, 1): ("С",  "Рок белгісі"),       # указательный + мизинец
    (1, 0, 0, 0, 1): ("О",  "Шаян"),              # большой + мизинец
    (0, 0, 0, 1, 1): ("П",  "Екі саусақ"),        # безымянный + мизинец
    (1, 0, 0, 1, 0): ("Ұ",  "Анонимді"),          # большой + безымянный
    # Три пальца
    (0, 1, 1, 1, 0): ("Г",  "Үш саусақ"),         # указ + средн + безым
    (1, 1, 1, 0, 0): ("М",  "Үш саусақ + б.б"),   # большой + указ + средн
    (0, 0, 1, 1, 1): ("Н",  "Үш оң саусақ"),      # средн + безым + мизинец
    (0, 1, 0, 1, 1): ("Ң",  "Аралас үш"),          # указ + безым + мизинец
    (1, 1, 0, 0, 1): ("Ы",  "Тарақ"),             # большой + указ + мизинец
    (1, 0, 1, 1, 0): ("І",  "Орта тарақ"),        # большой + средн + безым
    # Четыре пальца
    (1, 1, 1, 1, 0): ("Р",  "Төрт саусақ"),       # без мизинца
    (0, 1, 1, 1, 1): ("Т",  "Төрт оң"),           # без большого
    (1, 1, 0, 1, 1): ("Ш",  "Арнайы"),            # без среднего
    (1, 0, 1, 1, 1): ("Ш",  "Арнайы 2"),          # без указательного
    (1, 1, 1, 0, 1): ("Щ",  "Арнайы 3"),          # без безымянного
}


@dataclass
class GestureResult:
    """Результат классификации одного кадра."""
    letter:       str            # Буква КЖЯ ("А", "Ж", ...)
    description:  str            # Описание жеста на казахском
    finger_state: tuple          # (thumb, index, middle, ring, pinky)
    confidence:   float          # 0.0–1.0 (пока бинарная: 0 или 1)
    is_stable:    bool           # True если дебаунсинг пройден
    stable_frames: int           # Сколько кадров жест удерживается


class GestureEngine:
    """
    Геометрический классификатор жестов.

    Использование:
        engine = GestureEngine(min_stable_frames=20)
        result = engine.update(hand_landmarks)
        if result and result.is_stable:
            print(result.letter)
    """

    def __init__(self, min_stable_frames: int = 20):
        self.min_stable_frames = min_stable_frames

        # Дебаунс-история последних N классификаций
        self._history: deque[str] = deque(maxlen=min_stable_frames)
        self._stable_count:  int = 0
        self._last_committed: Optional[str] = None  # Последняя подтверждённая буква

    # ──────────────────────────────────────────
    # Публичный метод
    # ──────────────────────────────────────────

    def update(self, landmarks: list) -> Optional[GestureResult]:
        """
        Обновляет классификатор на основе landmarks текущего кадра.

        Args:
            landmarks: список из 21 объекта с полями .x, .y, .z (из MediaPipe)

        Returns:
            GestureResult или None, если жест не распознан
        """
        if landmarks is None or len(landmarks) < 21:
            self._reset()
            return None

        finger_state = self._get_finger_states(landmarks)
        letter, description = GESTURE_MAP.get(finger_state, (None, None))

        if letter is None:
            self._reset()
            return None

        # Дебаунсинг: добавляем в историю и считаем стабильные кадры
        self._history.append(letter)
        if len(self._history) == self.min_stable_frames and \
                len(set(self._history)) == 1:
            self._stable_count += 1
        else:
            self._stable_count = 0

        is_stable = self._stable_count >= 1  # стабилен с момента заполнения окна

        return GestureResult(
            letter=letter,
            description=description,
            finger_state=finger_state,
            confidence=1.0 if is_stable else 0.5,
            is_stable=is_stable,
            stable_frames=self._stable_count,
        )

    def is_new_gesture(self, result: Optional["GestureResult"]) -> bool:
        """
        Возвращает True, если стабильный жест — новый (не тот же, что уже добавили).
        Используется при добавлении буквы в транскрипцию.
        """
        if result is None or not result.is_stable:
            return False
        if result.letter != self._last_committed:
            return True
        return False

    def commit(self, letter: str) -> None:
        """Зафиксировать букву как добавленную в транскрипцию."""
        self._last_committed = letter

    def reset_commit(self) -> None:
        """Сбросить «последнюю добавленную» — разрешает снова добавить ту же букву."""
        self._last_committed = None

    # ──────────────────────────────────────────
    # Внутренние методы
    # ──────────────────────────────────────────

    def _reset(self) -> None:
        """Сбрасывает историю при потере руки."""
        self._history.clear()
        self._stable_count = 0

    def _get_finger_states(self, lm: list) -> tuple:
        """
        Определяет состояние 5 пальцев: разогнут (1) / согнут (0).
        Возвращает кортеж: (thumb, index, middle, ring, pinky)
        """
        thumb  = self._thumb_extended(lm)
        index  = self._finger_extended(lm, INDEX_TIP,  INDEX_PIP)
        middle = self._finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP)
        ring   = self._finger_extended(lm, RING_TIP,   RING_PIP)
        pinky  = self._finger_extended(lm, PINKY_TIP,  PINKY_PIP)
        return (thumb, index, middle, ring, pinky)

    @staticmethod
    def _finger_extended(lm: list, tip_idx: int, pip_idx: int) -> int:
        """
        Палец разогнут, если его кончик выше PIP-сустава (ось Y).
        В системе координат MediaPipe: меньше Y = выше на экране.
        """
        return int(lm[tip_idx].y < lm[pip_idx].y)

    @staticmethod
    def _thumb_extended(lm: list) -> int:
        """
        Большой палец: сравниваем расстояние кончика от INDEX_MCP
        с расстоянием IP-сустава от INDEX_MCP.
        Если кончик дальше — палец разогнут (работает для обеих рук).
        """
        def dist(a, b) -> float:
            return math.sqrt(
                (lm[a].x - lm[b].x) ** 2 +
                (lm[a].y - lm[b].y) ** 2
            )
        return int(dist(THUMB_TIP, INDEX_MCP) > dist(THUMB_IP, INDEX_MCP))
