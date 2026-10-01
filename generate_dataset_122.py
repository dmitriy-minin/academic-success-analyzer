"""
generate_dataset_122.py – Генерация реалистичного датасета на 122 студента колледжа
для дисциплины объемом 72 академических часа (24 ч теория, 24 ч практика, 24 ч производственное обучение).
"""

import csv
import math
import random

# Фиксация зерна случайности для воспроизводимости
random.seed(122)

OUTPUT_FILE = "dataset.csv"
BACKUP_FILE = "dataset_college_72h.csv"

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

def generate_student_record(cluster: str) -> dict:
    """
    Генерирует профиль студента в соответствии с его кластером успеваемости:
    - 'excellent': отличники (высокая мотивация, высокая посещаемость)
    - 'good': хорошисты (стабильные, регулярная работа)
    - 'average': удовлетворительно (средний темп, есть пропуски)
    - 'at_risk': группа риска (много пропусков, задолженности по практике)
    - 'critical': критический уровень (систематические прогулы, угроза отчисления)
    """
    if cluster == "excellent":
        num_grades = random.randint(18, 23)
        avg_grade = round(random.uniform(86.0, 98.5), 1)
        grade_trend = round(random.uniform(-1.5, 12.0), 1)
        num_absences = random.randint(0, 4)
        unexcused_absences = random.randint(0, min(1, num_absences))
        num_tardies = random.randint(0, 2)
        num_tests_completed = random.choice([5, 6, 6, 6])
        avg_time_on_tests = round(random.uniform(22.0, 34.0), 1)
        avg_test_attempts = round(random.uniform(1.0, 1.4), 1)
        num_assignments_submitted = random.choice([11, 12, 12])
        avg_time_on_assignments = round(random.uniform(55.0, 80.0), 1)

    elif cluster == "good":
        num_grades = random.randint(14, 19)
        avg_grade = round(random.uniform(73.0, 85.5), 1)
        grade_trend = round(random.uniform(-6.0, 10.0), 1)
        num_absences = random.randint(2, 7)
        unexcused_absences = random.randint(0, min(3, num_absences))
        num_tardies = random.randint(1, 5)
        num_tests_completed = random.choice([5, 6])
        avg_time_on_tests = round(random.uniform(25.0, 38.0), 1)
        avg_test_attempts = round(random.uniform(1.2, 1.8), 1)
        num_assignments_submitted = random.randint(9, 11)
        avg_time_on_assignments = round(random.uniform(60.0, 95.0), 1)

    elif cluster == "average":
        num_grades = random.randint(10, 15)
        avg_grade = round(random.uniform(58.0, 72.5), 1)
        grade_trend = round(random.uniform(-12.0, 8.0), 1)
        num_absences = random.randint(6, 13)
        unexcused_absences = random.randint(2, min(8, num_absences))
        num_tardies = random.randint(2, 7)
        num_tests_completed = random.randint(3, 5)
        avg_time_on_tests = round(random.uniform(18.0, 42.0), 1)
        avg_test_attempts = round(random.uniform(1.7, 2.5), 1)
        num_assignments_submitted = random.randint(6, 9)
        avg_time_on_assignments = round(random.uniform(45.0, 110.0), 1)

    elif cluster == "at_risk":
        num_grades = random.randint(6, 11)
        avg_grade = round(random.uniform(40.0, 57.5), 1)
        grade_trend = round(random.uniform(-18.0, 4.0), 1)
        num_absences = random.randint(13, 22)
        unexcused_absences = random.randint(8, min(17, num_absences))
        num_tardies = random.randint(4, 11)
        num_tests_completed = random.randint(1, 3)
        # либо быстро угадывал, либо долго сидел
        if random.random() < 0.6:
            avg_time_on_tests = round(random.uniform(9.0, 19.0), 1)
        else:
            avg_time_on_tests = round(random.uniform(38.0, 46.0), 1)
        avg_test_attempts = round(random.uniform(2.1, 3.2), 1)
        num_assignments_submitted = random.randint(2, 5)
        avg_time_on_assignments = round(random.uniform(30.0, 140.0), 1)

    else: # critical
        num_grades = random.randint(2, 6)
        avg_grade = round(random.uniform(22.0, 39.5), 1)
        grade_trend = round(random.uniform(-25.0, -2.0), 1)
        num_absences = random.randint(20, 30)
        unexcused_absences = random.randint(16, num_absences)
        num_tardies = random.randint(6, 14)
        num_tests_completed = random.choice([0, 1, 1, 2])
        avg_time_on_tests = round(random.uniform(6.0, 16.0), 1)
        avg_test_attempts = round(random.uniform(2.0, 3.5), 1)
        num_assignments_submitted = random.choice([0, 1, 2])
        avg_time_on_assignments = round(random.uniform(15.0, 45.0), 1)

    # Расчет рекомендуемой итоговой оценки target_grade (0-100)
    # Сумма положительных компонентов = 100%
    score = 0.0

    # 1. Текущий средний балл за работы (вес 45%)
    score += 0.45 * avg_grade

    # 2. Выполнение практических и производственных заданий (12 заданий, вес 25%)
    assign_ratio = clamp(num_assignments_submitted / 12.0, 0.0, 1.0)
    score += 0.25 * assign_ratio * 100.0

    # 3. Прохождение тестов по теории (6 тестов, вес 15%)
    test_ratio = clamp(num_tests_completed / 6.0, 0.0, 1.0)
    score += 0.15 * test_ratio * 100.0

    # 4. Динамика успеваемости / тренд (вес 7%)
    # Нормализуем тренд от -25 до +25 в [0, 1]
    trend_norm = clamp((grade_trend + 20.0) / 40.0, 0.0, 1.0)
    score += 0.07 * trend_norm * 100.0

    # 5. Эффективность прохождения тестов (число попыток, вес 4%)
    attempt_ratio = clamp((3.2 - avg_test_attempts) / 2.2, 0.0, 1.0)
    score += 0.04 * attempt_ratio * 100.0

    # 6. Рациональность времени на задания (вес 4%)
    opt_time = 70.0
    time_delta = abs(avg_time_on_assignments - opt_time)
    time_score = clamp(1.0 - time_delta / 80.0, 0.0, 1.0)
    score += 0.04 * time_score * 100.0

    # Штраф за пропуски занятий (всего 36 занятий по 2 ч = 72 ч)
    # Неуважительные пропуски наказываются строже уважительных
    excused_absences = max(0, num_absences - unexcused_absences)
    absence_penalty = (excused_absences * 0.25 + unexcused_absences * 0.75)
    score -= clamp(absence_penalty, 0.0, 18.0)

    # Штраф за опоздания
    score -= clamp(num_tardies * 0.2, 0.0, 3.0)

    # Добавление небольшого гауссовского шума для реалистичности
    score += random.gauss(0, 1.5)

    target_grade = clamp(round(score, 1), 0.0, 100.0)

    return {
        "num_grades": num_grades,
        "avg_grade": avg_grade,
        "grade_trend": grade_trend,
        "num_absences": num_absences,
        "unexcused_absences": unexcused_absences,
        "num_tardies": num_tardies,
        "num_tests_completed": num_tests_completed,
        "avg_time_on_tests": avg_time_on_tests,
        "avg_test_attempts": avg_test_attempts,
        "num_assignments_submitted": num_assignments_submitted,
        "avg_time_on_assignments": avg_time_on_assignments,
        "target_grade": target_grade,
    }


def generate_all_students(total_count: int = 122):
    """
    Генерирует точно total_count студентов с реалистичным распределением успеваемости в колледже:
    - Отличники (excellent): ~20%  (24 студента)
    - Хорошисты (good):      ~34%  (42 студента)
    - Троечники (average):   ~26%  (32 студента)
    - Группа риска (at_risk): ~13% (16 студентов)
    - Критический (critical): ~7%  (8 студентов)
    Итого: 24 + 42 + 32 + 16 + 8 = 122 студента
    """
    counts = {
        "excellent": 24,
        "good": 42,
        "average": 32,
        "at_risk": 16,
        "critical": 8,
    }
    assert sum(counts.values()) == total_count

    students = []
    for cluster, count in counts.items():
        for _ in range(count):
            students.append(generate_student_record(cluster))

    # Перемешиваем список, чтобы кластеры не шли подряд
    random.shuffle(students)
    return students


def main():
    students = generate_all_students(122)
    fieldnames = list(students[0].keys())

    for filename in [OUTPUT_FILE, BACKUP_FILE]:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(students)
        print(f"[OK] File '{filename}' successfully saved ({len(students)} students).")

    grades = [s["target_grade"] for s in students]
    avg_grades = [s["avg_grade"] for s in students]
    absences = [s["num_absences"] for s in students]

    print("\n--- Summary Statistics (122 college students, 72 hours) ---")
    print(f"Target Grade (target_grade):")
    print(f"  Min  : {min(grades):.1f}")
    print(f"  Max  : {max(grades):.1f}")
    print(f"  Mean : {sum(grades)/len(grades):.1f}")
    print(f"Current Average Grade (avg_grade): {sum(avg_grades)/len(avg_grades):.1f}")
    print(f"Average Absences: {sum(absences)/len(absences):.1f} pairs (out of 36)")
    
    # Категории оценок
    c_5 = sum(1 for g in grades if g >= 85.0)
    c_4 = sum(1 for g in grades if 70.0 <= g < 85.0)
    c_3 = sum(1 for g in grades if 55.0 <= g < 70.0)
    c_2 = sum(1 for g in grades if g < 55.0)
    print("\nGrade Category Distribution:")
    print(f"  Excellent (85-100)      : {c_5:2d} students ({c_5/len(grades)*100:.1f}%)")
    print(f"  Good (70-84.9)          : {c_4:2d} students ({c_4/len(grades)*100:.1f}%)")
    print(f"  Satisfactory (55-69.9)  : {c_3:2d} students ({c_3/len(grades)*100:.1f}%)")
    print(f"  At Risk / Fail (< 55)   : {c_2:2d} students ({c_2/len(grades)*100:.1f}%)")


if __name__ == "__main__":
    main()
