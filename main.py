"""
SignQazaq Pro — Портативный переводчик казахского жестового языка (КЖЯ)
Версия: 0.4.0

Pipeline:
  Камера → MediaPipe HandLandmarker → GestureEngine → TranscriptionBuffer → TTSEngine

Управление:
  SPACE  / жест «пробел» (1.5с пауза) — добавить пробел
  ENTER  — прочитать всю транскрипцию вслух
  ←      — удалить последний символ
  C      — очистить транскрипцию
  ESC/Q  — выход
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time
import urllib.request

from gesture_engine import GestureEngine, GestureResult, GESTURE_MAP
from tts_engine import TTSEngine

# ─────────────────────────────────────────────
# Автоматическая загрузка модели MediaPipe
# ─────────────────────────────────────────────
MODEL_PATH = "models/hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
if not os.path.exists(MODEL_PATH):
    print("📥 Скачиваю модель (~28 МБ)...")
    os.makedirs("models", exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Модель сохранена:", MODEL_PATH)

# ─────────────────────────────────────────────
# Константы
# ─────────────────────────────────────────────
FRAME_W  = 640
FRAME_H  = 480

# Дебаунсинг жестов: сколько кадров держать жест для подтверждения
MIN_STABLE_FRAMES = 22

# Через сколько секунд «нет руки» → автоматический пробел в транскрипции
NO_HAND_PAUSE_SEC = 1.5

# Максимальная длина строки транскрипции
MAX_TRANSCRIPT_LEN = 80

# Соединения руки для ручной отрисовки
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# Палитра цветов UI
CLR_GREEN   = (0,  255, 128)
CLR_YELLOW  = (255, 200, 0)
CLR_CYAN    = (0,  200, 255)
CLR_WHITE   = (255, 255, 255)
CLR_GRAY    = (140, 140, 140)
CLR_BLACK   = (0,   0,   0)
CLR_ACCENT  = (80,  120, 255)   # Синий акцент
CLR_CONFIRM = (50,  230, 100)   # Зелёный при подтверждении


# ─────────────────────────────────────────────
# Инициализация MediaPipe Tasks API
# ─────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
)


# ─────────────────────────────────────────────
# Отрисовка landmarks руки
# ─────────────────────────────────────────────

def draw_hand(frame, landmarks, handedness, w, h, result: GestureResult):
    """Рисует скелет руки. Цвет = статус распознавания."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # Цвет зависит от стабильности жеста
    clr_joint = CLR_CONFIRM if (result and result.is_stable) else CLR_GREEN
    clr_conn  = CLR_YELLOW  if (result and result.is_stable) else (200, 200, 0)

    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], clr_conn, 2, cv2.LINE_AA)

    for i, pt in enumerate(pts):
        r = 6 if i == 0 else 4
        cv2.circle(frame, pt, r, clr_joint, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, r, CLR_BLACK, 1,  cv2.LINE_AA)

    # Подпись Оң қол / Сол қол
    side = handedness[0].category_name
    label = "Оң қол" if side == "Right" else "Сол қол"
    wx, wy = pts[0]
    cv2.putText(frame, label, (wx - 35, wy - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_WHITE, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────
# Отрисовка UI блоков
# ─────────────────────────────────────────────

def draw_top_bar(frame, fps, hand_count, voice_name, w):
    """Верхняя полоса: FPS / руки / голос."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), CLR_BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"FPS {fps:5.1f}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, CLR_GREEN, 2)
    cv2.putText(frame, f"Рук: {hand_count}",
                (150, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, CLR_CYAN, 2)
    cv2.putText(frame, f"🔊 {voice_name}",
                (260, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLR_GRAY, 1)


def draw_gesture_box(frame, result: GestureResult, flash_alpha: float, w, h):
    """
    Центральный блок текущего жеста.
    flash_alpha: 0.0–1.0, вспышка зелёного при подтверждении.
    """
    # Положение блока
    box_w, box_h = 140, 140
    bx = w // 2 - box_w // 2
    by = h // 2 - box_h // 2 + 20

    if result is None:
        # Нет жеста — серый контур
        cv2.rectangle(frame, (bx, by), (bx+box_w, by+box_h), (60,60,60), 2)
        cv2.putText(frame, "?", (bx+54, by+98),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (70,70,70), 3, cv2.LINE_AA)
        return

    # Цвет рамки: синий (нестабильный) → зелёный (стабильный/вспышка)
    if flash_alpha > 0:
        r = int(50  + (50  - 50)  * flash_alpha)
        g = int(120 + (230 - 120) * flash_alpha)
        b = int(255 + (100 - 255) * flash_alpha)
        box_clr = (b, g, r)  # BGR
    elif result.is_stable:
        box_clr = CLR_CONFIRM
    else:
        box_clr = CLR_ACCENT

    # Толщина рамки по стабильности
    thickness = 3 if result.is_stable else 2

    # Полупрозрачный фон
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx+box_w, by+box_h), (20, 20, 40), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx+box_w, by+box_h), box_clr, thickness)

    # Буква (крупно, по центру)
    letter = result.letter
    font_scale = 3.2 if len(letter) == 1 else 2.0
    (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 4)
    tx = bx + (box_w - tw) // 2
    ty = by + (box_h + th) // 2 - 8
    cv2.putText(frame, letter, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, CLR_WHITE, 4, cv2.LINE_AA)

    # Прогресс-бар стабильности (внизу блока)
    progress = min(result.stable_frames / MIN_STABLE_FRAMES, 1.0)
    bar_x1, bar_y = bx + 6, by + box_h - 8
    bar_len = int((box_w - 12) * progress)
    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + box_w - 12, bar_y + 4),
                  (50, 50, 50), -1)
    if bar_len > 0:
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + bar_len, bar_y + 4),
                      box_clr, -1)

    # Описание жеста (под блоком)
    desc = result.description
    (dw, _), _ = cv2.getTextSize(desc, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    dx = bx + (box_w - dw) // 2
    cv2.putText(frame, desc, (dx, by + box_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_GRAY, 1, cv2.LINE_AA)


def draw_finger_debug(frame, result: GestureResult, x, y):
    """Мини-отладчик: показывает состояние 5 пальцев."""
    if result is None:
        return
    names = ["T", "I", "M", "R", "P"]
    for i, (name, val) in enumerate(zip(names, result.finger_state)):
        clr = CLR_GREEN if val else CLR_GRAY
        cv2.circle(frame, (x + i * 22, y), 8, clr, -1, cv2.LINE_AA)
        cv2.putText(frame, name, (x + i * 22 - 5, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLR_BLACK, 1)


def draw_transcription(frame, text: str, is_reading: bool, w, h):
    """Нижняя панель транскрипции."""
    panel_h = 80
    py = h - panel_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, py), (w, h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Разделительная линия
    line_clr = CLR_CONFIRM if is_reading else (60, 60, 80)
    cv2.line(frame, (0, py), (w, py), line_clr, 1)

    # Заголовок
    cv2.putText(frame, "ТРАНСКРИПЦИЯ",
                (10, py + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, CLR_GRAY, 1)

    # Текст транскрипции (с курсором)
    display = (text or "—") + ("▌" if not is_reading else "")
    # Обрезаем с конца если длинный
    if len(display) > 45:
        display = "..." + display[-42:]

    font_scale = 1.0
    clr = CLR_CONFIRM if is_reading else CLR_WHITE
    cv2.putText(frame, display,
                (10, py + 55), cv2.FONT_HERSHEY_SIMPLEX, font_scale, clr, 2, cv2.LINE_AA)

    # Подсказки клавиш
    hints = "[ENTER] читать  [C] очистить  [←] удалить  [SPACE] пробел"
    cv2.putText(frame, hints,
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 100), 1)


def draw_flash(frame, alpha: float, w, h):
    """Вспышка зелёного при добавлении буквы."""
    if alpha <= 0:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), CLR_CONFIRM, -1)
    cv2.addWeighted(overlay, alpha * 0.18, frame, 1 - alpha * 0.18, 0, frame)


# ─────────────────────────────────────────────
# Основной цикл
# ─────────────────────────────────────────────

def main():
    # Инициализация подсистем
    gesture_engine = GestureEngine(min_stable_frames=MIN_STABLE_FRAMES)
    tts = TTSEngine(rate=155)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        print("❌ Камера не найдена!")
        return

    print("=" * 52)
    print("  SignQazaq Pro v0.4 — Запущен ✅")
    print("=" * 52)
    print("  Покажи жест перед камерой")
    print("  [ENTER] читать | [C] очистить | [ESC] выход")
    print("=" * 52)

    # Состояние приложения
    transcript:       str   = ""       # Накопленная транскрипция
    fps:              float = 0.0
    prev_time:        float = time.time()
    frame_ts_ms:      int   = 0        # Временная метка для Tasks API
    flash_alpha:      float = 0.0      # Вспышка при добавлении буквы
    no_hand_since:    float = time.time()  # Когда пропала рука
    space_added:      bool  = False    # Пробел уже добавлен за эту паузу
    is_reading:       bool  = False    # TTS читает фразу
    current_result: GestureResult | None = None

    with HandLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # ── FPS ──
            t = time.time()
            fps = 1.0 / (t - prev_time + 1e-9)
            prev_time = t

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # ── MediaPipe детекция ──
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts_ms += 33
            result = detector.detect_for_video(mp_img, frame_ts_ms)

            has_hand   = bool(result.hand_landmarks)
            hand_count = len(result.hand_landmarks) if has_hand else 0

            # ── Жест: детект и классификация ──
            current_result = None
            if has_hand:
                no_hand_since = t
                space_added   = False

                # Берём первую руку
                lm        = result.hand_landmarks[0]
                handedness = result.handedness[0]

                current_result = gesture_engine.update(lm)

                # Рисуем скелет
                draw_hand(frame, lm, handedness, w, h, current_result)

                # Если жест стабилен и новый → добавить в транскрипцию
                if gesture_engine.is_new_gesture(current_result):
                    letter = current_result.letter
                    if len(transcript) < MAX_TRANSCRIPT_LEN:
                        transcript += letter
                    gesture_engine.commit(letter)
                    tts.speak_letter(letter)   # Озвучиваем букву
                    flash_alpha = 1.0          # Запускаем вспышку
                    print(f"  ✍️  [{letter}]  →  «{transcript}»")

                # Если рук нет — сброс, чтобы та же буква снова сработала
                gesture_engine.reset_commit() if not has_hand else None

            else:
                gesture_engine.update(None)  # Сбросить дебаунсер

                # Авто-пробел после паузы NO_HAND_PAUSE_SEC
                if not space_added and (t - no_hand_since) > NO_HAND_PAUSE_SEC:
                    if transcript and not transcript.endswith(" "):
                        transcript += " "
                        space_added = True
                        print("  ⎵  Пробел")

            # ── Отрисовка UI ──
            draw_flash(frame, flash_alpha, w, h)
            draw_top_bar(frame, fps, hand_count, tts.default_voice, w)
            draw_gesture_box(frame, current_result, flash_alpha, w, h)

            if current_result:
                draw_finger_debug(frame, current_result, w - 130, 26)

            is_reading = tts.is_speaking
            draw_transcription(frame, transcript, is_reading, w, h)

            # Затухание вспышки
            if flash_alpha > 0:
                flash_alpha = max(0.0, flash_alpha - 0.07)

            cv2.imshow("SignQazaq Pro v0.4", frame)

            # ── Клавиатурные команды ──
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord('q')):          # ESC / Q — выход
                break

            elif key == 13:                    # ENTER — читать всю транскрипцию
                if transcript.strip():
                    print(f"  🔊 Читаю: «{transcript.strip()}»")
                    tts.speak_phrase(transcript.strip())

            elif key == ord('c') or key == ord('C'):  # C — очистить
                transcript = ""
                gesture_engine.reset_commit()
                print("  🗑  Транскрипция очищена")

            elif key == 8 or key == 127:       # Backspace — удалить символ
                transcript = transcript[:-1]
                print(f"  ⬅  Удалено. Сейчас: «{transcript}»")

            elif key == ord(' '):              # SPACE — принудительный пробел
                if transcript and not transcript.endswith(" "):
                    transcript += " "

    # ── Завершение ──
    cap.release()
    cv2.destroyAllWindows()
    tts.stop()
    print(f"\n  Завершено. Итоговая транскрипция: «{transcript.strip()}»")


if __name__ == "__main__":
    main()