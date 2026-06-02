"""
model.py – Определение архитектуры нейронной сети для аналитики успеваемости студентов.

Нейронная сеть принимает 11 числовых признаков и возвращает рекомендуемую оценку (0–100).
"""

import torch
import torch.nn as nn


# Имена входных признаков (для удобства)
FEATURE_NAMES = [
    "num_grades",            # Количество оценок
    "avg_grade",             # Средняя оценка
    "grade_trend",           # Тренд (разница между оценками)
    "num_absences",          # Количество пропусков
    "unexcused_absences",    # Неуважительные пропуски
    "num_tardies",           # Количество опозданий
    "num_tests_completed",   # Количество пройденных тестов
    "avg_time_on_tests",     # Среднее время на тест (мин)
    "avg_test_attempts",     # Среднее количество попыток на тест
    "num_assignments_submitted",  # Количество сданных заданий
    "avg_time_on_assignments",    # Среднее время на задание (мин)
]

INPUT_SIZE  = len(FEATURE_NAMES)   # 11 входных нейронов
OUTPUT_SIZE = 1                     # 1 выходной нейрон – рекомендуемая оценка


class StudentPerformanceNet(nn.Module):
    """
    Полносвязная нейронная сеть с тремя скрытыми слоями.

    Архитектура:
        Вход (11) → Слой 1 (64) → BatchNorm → ReLU → Dropout
                  → Слой 2 (128) → BatchNorm → ReLU → Dropout
                  → Слой 3 (64)  → BatchNorm → ReLU → Dropout
                  → Слой 4 (32)  → ReLU
                  → Выход (1)    → Sigmoid × 100

    Выходное значение находится в диапазоне [0, 100] – рекомендуемая оценка.
    """

    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()

        self.network = nn.Sequential(
            # Слой 1: 11 - 64
            nn.Linear(INPUT_SIZE, 64),
            nn.BatchNorm1d(64),   # нормализация для стабильного обучения
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Слой 2: 64 - 128
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Слой 3: 128 - 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Слой 4: 64 - 32
            nn.Linear(64, 32),
            nn.ReLU(),

            # Выходной слой: 32 - 1
            nn.Linear(32, OUTPUT_SIZE),
            nn.Sigmoid(),   # выход в [0, 1], затем масштабируем на 100
        )

        # Инициализация весов методом Xavier для лучшей сходимости
        self._init_weights()

    def _init_weights(self):
        """Инициализирует линейные слои весами Xavier."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход сети.

        Args:
            x: Тензор формы (batch_size, 11) с нормализованными признаками.

        Returns:
            Тензор формы (batch_size, 1) – рекомендуемая оценка в диапазоне [0, 100].
        """
        return self.network(x) * 100.0   # масштабирует Sigmoid → [0, 100]
