"""
generate_dataset.py – Генерация синтетического обучающего датасета (100 примеров).

Каждая строка описывает одного студента. Целевая оценка рассчитывается по
детерминированной формуле с добавлением небольшого шума, что имитирует
реальные данные.

Запуск:
    python generate_dataset.py
"""

import csv
import random
import math
import os

# Фиксирует случайное зерно для воспроизводимости
random.seed(42)

OUTPUT_FILE = "dataset.csv"

# Вспомогательные функции
def clamp(value: float, lo: float, hi: float) -> float:
    """Ограничивает значение диапазоном [lo, hi]."""
    return max(lo, min(hi, value))


def generate_student() -> dict:
    """
    Генерирует случайного студента с реалистичными значениями признаков.
    Возвращает словарь: признаки + целевая оценка.
    """
    # Академические показатели
    num_grades = random.randint(5, 30)
    avg_grade  = round(random.uniform(40.0, 100.0), 1)
    # Тренд: положительный – улучшение, отрицательный – ухудшение
    grade_trend = round(random.uniform(-20.0, 20.0), 1)

    # Посещаемость
    num_absences       = random.randint(0, 20)
    unexcused_absences = random.randint(0, num_absences)   # не больше общего числа
    num_tardies        = random.randint(0, 15)

    # Тесты
    num_tests_completed = random.randint(0, 15)
    avg_time_on_tests   = round(random.uniform(5.0, 90.0), 1)   # минуты
    avg_test_attempts   = round(random.uniform(1.0, 5.0), 1)

    # Задания
    num_assignments_submitted  = random.randint(0, 20)
    avg_time_on_assignments    = round(random.uniform(10.0, 180.0), 1)  # минуты

    # Вычисление целевой оценки
    # Формула учитывает вклад каждого признака в итоговую успеваемость:
    score = 0.0

    # 1. Средняя оценка – самый весомый вклад (40 %)
    score += 0.40 * avg_grade

    # 2. Тренд – студент с улучшающимися результатами получает бонус
    score += 0.10 * (grade_trend + 20.0) / 40.0 * 100.0

    # 3. Пропуски снижают оценку (штраф до −15 баллов)
    absence_penalty = clamp(num_absences * 1.2 + unexcused_absences * 1.5, 0, 25)
    score -= 0.15 * absence_penalty

    # 4. Опоздания – небольшой штраф (до −5 баллов)
    tardy_penalty = clamp(num_tardies * 0.5, 0, 5)
    score -= 0.05 * tardy_penalty * 10

    # 5. Пройденные тесты – бонус за активность (до +10 %)
    test_completion_ratio = clamp(num_tests_completed / 15.0, 0, 1)
    score += 0.10 * test_completion_ratio * 100.0

    # 6. Среднее число попыток – меньше попыток = лучше
    attempt_bonus = clamp((5.0 - avg_test_attempts) / 4.0, 0, 1)
    score += 0.05 * attempt_bonus * 100.0

    # 7. Сданные задания – бонус за выполнение
    assignment_ratio = clamp(num_assignments_submitted / 20.0, 0, 1)
    score += 0.15 * assignment_ratio * 100.0

    # 8. Время на задания – умеренное время считается оптимальным
    # Слишком мало (не старается) или слишком много (неэффективен) – штраф
    optimal_time = 60.0  # минут
    time_diff = abs(avg_time_on_assignments - optimal_time)
    time_bonus = clamp(1.0 - time_diff / 120.0, 0, 1)
    score += 0.05 * time_bonus * 100.0

    # Добавляет шум ±3 балла для реалистичности
    score += random.gauss(0, 3.0)

    # Финальное ограничение диапазона
    score = clamp(round(score, 1), 0.0, 100.0)

    return {
        "num_grades":                 num_grades,
        "avg_grade":                  avg_grade,
        "grade_trend":                grade_trend,
        "num_absences":               num_absences,
        "unexcused_absences":         unexcused_absences,
        "num_tardies":                num_tardies,
        "num_tests_completed":        num_tests_completed,
        "avg_time_on_tests":          avg_time_on_tests,
        "avg_test_attempts":          avg_test_attempts,
        "num_assignments_submitted":  num_assignments_submitted,
        "avg_time_on_assignments":    avg_time_on_assignments,
        "target_grade":               score,   # целевое значение
    }


def main():
    """Генерирует 100 примеров и сохраняет их в CSV-файл."""
    students = [generate_student() for _ in range(100)]

    fieldnames = list(students[0].keys())

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(students)

    print(f"✅ Датасет сохранён в «{OUTPUT_FILE}» ({len(students)} записей).")

    # Краткая статистика по целевым оценкам
    grades = [s["target_grade"] for s in students]
    print(f"   Мин. оценка : {min(grades):.1f}")
    print(f"   Макс. оценка: {max(grades):.1f}")
    print(f"   Средн. оценка: {sum(grades)/len(grades):.1f}")


if __name__ == "__main__":
    main()
