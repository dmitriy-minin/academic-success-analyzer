"""
dataset.py – Утилиты для загрузки, разбивки и нормализации датасета.

Функции:
    load_dataset  – читает CSV, возвращает (X, y) как numpy-массивы
    split_dataset – делит на train/val/test
    normalize_features – StandardScaler с fit только на train
"""

import csv
import numpy as np
from typing import Tuple

from model import FEATURE_NAMES



# 1. Загрузка CSV
def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Читает CSV-файл датасета и возвращает матрицу признаков X и вектор меток y.

    Args:
        filepath: Путь к CSV-файлу (например, «dataset.csv»).

    Returns:
        X: numpy-массив формы (N, 11) – входные признаки.
        y: numpy-массив формы (N,)   – целевые оценки [0, 100].
    """
    X_rows, y_rows = [], []

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Извлекает признаки в строгом порядке из FEATURE_NAMES
            features = [float(row[name]) for name in FEATURE_NAMES]
            X_rows.append(features)
            y_rows.append(float(row["target_grade"]))

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)



# 2. Разбивка на выборки
def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed: int = 42,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Делит данные на обучающую, валидационную и тестовую выборки.

    Args:
        X:           Матрица признаков.
        y:           Вектор меток.
        train_ratio: Доля обучающей выборки (по умолчанию 70 %).
        val_ratio:   Доля валидационной выборки (по умолчанию 15 %).
        seed:        Зерно генератора случайных чисел.

    Returns:
        Три пары (X_split, y_split) для train, val, test.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    n_train = int(len(X) * train_ratio)
    n_val   = int(len(X) * val_ratio)

    idx_train = indices[:n_train]
    idx_val   = indices[n_train : n_train + n_val]
    idx_test  = indices[n_train + n_val :]

    return (
        (X[idx_train], y[idx_train]),
        (X[idx_val],   y[idx_val]),
        (X[idx_test],  y[idx_test]),
    )



# 3. Нормализация признаков (StandardScaler)
class StandardScaler:
    """
    Простая реализация StandardScaler:
        z = (x − mean) / std
    Параметры (mean, std) вычисляются только на обучающей выборке.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_:  np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Вычисляет среднее и стандартное отклонение по обучающим данным."""
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        # Защита от деления на ноль (если признак константный)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Применяет нормализацию."""
        if self.mean_ is None:
            raise RuntimeError("Сначала вызовите fit().")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Удобный метод: fit + transform в одном вызове."""
        return self.fit(X).transform(X)


def normalize_features(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Нормализует все три выборки. Параметры считаются только по X_train.

    Returns:
        Нормализованные X_train, X_val, X_test и обученный скейлер.
    """
    scaler = StandardScaler()
    X_train_n = scaler.fit_transform(X_train)
    X_val_n   = scaler.transform(X_val)
    X_test_n  = scaler.transform(X_test)
    return X_train_n, X_val_n, X_test_n, scaler
