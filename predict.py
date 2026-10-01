"""
predict.py – Инференс: предсказание рекомендуемой оценки для студента.

Использование:
    # Интерактивный ввод (задаёт вопросы)
    python predict.py

    # Передача данных через аргументы
    python predict.py --input "20 78.5 3.2 4 1 2 10 35.0 1.5 18 55.0"

    # Пакетное предсказание из CSV (без колонки target_grade)
    python predict.py --batch students_new.csv
"""

import argparse
import pickle
import sys

import numpy as np
import torch

from model import StudentPerformanceNet, FEATURE_NAMES


# Загрузка модели и нормализатора
def load_model(model_path: str, scaler_path: str):
    """
    Загружает обученную модель и нормализатор.

    Returns:
        model:  StudentPerformanceNet в режиме eval
        scaler: обученный StandardScaler
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = StudentPerformanceNet(dropout_rate=checkpoint.get("dropout_rate", 0.2))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


# Предсказание для одного студента
def predict_one(model, scaler, features: list) -> float:
    """
    Вычисляет рекомендуемую оценку для одного студента.

    Args:
        model:    Обученная нейросеть.
        scaler:   Нормализатор.
        features: Список из 11 числовых значений признаков.

    Returns:
        Рекомендуемая оценка в диапазоне [0.0, 100.0].
    """
    X = np.array(features, dtype=np.float32).reshape(1, -1)
    X_norm = scaler.transform(X)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)

    with torch.no_grad():
        output = model(X_tensor)

    return float(output.item())


# Интерактивный ввод
# Подробные подсказки для каждого признака
PROMPTS = [
    ("Количество оценок",                "целое, например: 20"),
    ("Средняя оценка",                   "от 0 до 100, например: 75.5"),
    ("Тренд (разница между оценками)",   "например: 3.0 (рост) или -2.0 (снижение)"),
    ("Количество пропусков",             "целое, например: 5"),
    ("Неуважительные пропуски",          "целое, ≤ числу всех пропусков"),
    ("Количество опозданий",             "целое, например: 3"),
    ("Количество пройденных тестов",     "целое, например: 10"),
    ("Среднее время на тест (мин)",      "например: 40.0"),
    ("Среднее число попыток на тест",    "например: 1.5"),
    ("Количество сданных заданий",       "целое, например: 18"),
    ("Среднее время на задание (мин)",   "например: 60.0"),
]


def interactive_input() -> list:
    """Запрашивает значения признаков у пользователя в интерактивном режиме."""
    print("\nВведите данные студента:")
    print("-" * 50)
    values = []
    for (label, hint), name in zip(PROMPTS, FEATURE_NAMES):
        while True:
            try:
                raw = input(f"  {label} ({hint}): ").strip()
                values.append(float(raw))
                break
            except ValueError:
                print("    [!] Введите числовое значение.")
    return values


# Пакетный режим (CSV)

def batch_predict(model, scaler, csv_path: str):
    """Читает CSV (без колонки target_grade) и выводит предсказания."""
    import csv

    print(f"\nПакетное предсказание из '{csv_path}'")
    print("-" * 60)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            features = [float(row[name]) for name in FEATURE_NAMES]
            grade = predict_one(model, scaler, features)
            print(f"  Студент {i:3d} → Рекомендуемая оценка: {grade:6.1f}")


# Интерпретация оценки
def interpret(grade: float) -> str:
    """Возвращает текстовую интерпретацию числовой оценки."""
    if grade >= 90:
        return "Отлично"
    elif grade >= 75:
        return "Хорошо"
    elif grade >= 60:
        return "Удовлетворительно"
    else:
        return "Неудовлетворительно"


# Точка входа
def main():
    parser = argparse.ArgumentParser(description="Предсказание оценки студента")
    parser.add_argument("--model",   default="model.pth",   help="Путь к файлу модели")
    parser.add_argument("--scaler",  default="scaler.pkl",  help="Путь к нормализатору")
    parser.add_argument("--input",   default=None,
                        help="11 чисел через пробел (значения признаков)")
    parser.add_argument("--batch",   default=None,
                        help="CSV-файл для пакетного предсказания")
    args = parser.parse_args()

    # Загружает модель
    try:
        model, scaler = load_model(args.model, args.scaler)
    except FileNotFoundError as e:
        print(f"❌ Файл не найден: {e}")
        print("   Сначала запустите: python train.py")
        sys.exit(1)

    # Пакетный режим
    if args.batch:
        batch_predict(model, scaler, args.batch)
        return

    # Режим одиночного предсказания
    if args.input:
        try:
            features = [float(v) for v in args.input.split()]
            assert len(features) == len(FEATURE_NAMES)
        except (ValueError, AssertionError):
            print(f"❌ Ожидается ровно {len(FEATURE_NAMES)} числовых значений.")
            sys.exit(1)
    else:
        features = interactive_input()

    # Предсказание
    grade = predict_one(model, scaler, features)

    # Красивый вывод
    print("\n" + "=" * 50)
    print(f"  Рекомендуемая оценка: {grade:.1f} / 100")
    print(f"  {interpret(grade)}")
    print("=" * 50)

    # Подробный вывод признаков
    print("\n  Введённые данные:")
    for name, val in zip(FEATURE_NAMES, features):
        print(f"    {name:35s}: {val}")


if __name__ == "__main__":
    main()
