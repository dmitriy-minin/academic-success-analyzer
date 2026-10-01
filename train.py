"""
train.py – Обучение нейронной сети на датасете успеваемости студентов.

Что делает скрипт:
    1. Загружает датасет из dataset.csv
    2. Нормализует признаки (StandardScaler)
    3. Делит данные на обучающую/валидационную/тестовую выборки
    4. Обучает StudentPerformanceNet
    5. Сохраняет обученную модель и нормализатор

Запуск:
    python train.py
    python train.py --epochs 200 --lr 0.001 --batch-size 16
"""

import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import StudentPerformanceNet, FEATURE_NAMES
from dataset import load_dataset, split_dataset, normalize_features


# Аргументы командной строки
def parse_args():
    parser = argparse.ArgumentParser(description="Обучение нейросети успеваемости")
    parser.add_argument("--dataset",    default="dataset_college_72h.csv", help="Путь к CSV-файлу")
    parser.add_argument("--epochs",     type=int,   default=150,  help="Число эпох")
    parser.add_argument("--lr",         type=float, default=0.001, help="Скорость обучения")
    parser.add_argument("--batch-size", type=int,   default=16,   help="Размер батча")
    parser.add_argument("--dropout",    type=float, default=0.2,  help="Вероятность Dropout")
    parser.add_argument("--model-out",  default="model.pth",      help="Файл модели")
    parser.add_argument("--scaler-out", default="scaler.pkl",     help="Файл нормализатора")
    return parser.parse_args()


# Обучение одной эпохи
def train_epoch(model, loader, optimizer, criterion, device):
    """Проходит по всему обучающему загрузчику, обновляет веса."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()           # обнуляет градиенты
        predictions = model(X_batch)    # прямой проход
        loss = criterion(predictions, y_batch)  # считает ошибку
        loss.backward()                 # обратное распространение
        optimizer.step()                # шаг оптимизатора

        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


# Оценка на валидационной/тестовой выборке
def evaluate(model, loader, criterion, device):
    """Возвращает среднюю ошибку MSE и MAE на заданном наборе данных."""
    model.eval()
    total_loss = 0.0
    total_mae  = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            total_loss += criterion(predictions, y_batch).item() * len(X_batch)
            total_mae  += torch.abs(predictions - y_batch).sum().item()

    n = len(loader.dataset)
    return total_loss / n, total_mae / n


# Основной процесс обучения
def main():
    args = parse_args()

    # Устройство: GPU если доступно, иначе CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    # 1. Загрузка данных
    print(f"Загрузка датасета из «{args.dataset}»…")
    X, y = load_dataset(args.dataset)
    print(f"   Загружено {len(X)} примеров, {X.shape[1]} признаков")

    # 2. Разбивка: 70 % обучение / 15 % валидация / 15 % тест
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(X, y)
    print(f"   Обучение: {len(X_train)}, Валидация: {len(X_val)}, Тест: {len(X_test)}")

    # 3. Нормализация признаков (fit только на train)
    X_train_n, X_val_n, X_test_n, scaler = normalize_features(X_train, X_val, X_test)

    # Сохраняет нормализатор – понадобится при инференсе
    with open(args.scaler_out, "wb") as f:
        pickle.dump(scaler, f)
    print(f"   Нормализатор сохранён в «{args.scaler_out}»")

    # 4. Создание тензоров и загрузчиков
    def to_loader(X_np, y_np, shuffle=False):
        X_t = torch.tensor(X_np, dtype=torch.float32)
        y_t = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
        return DataLoader(TensorDataset(X_t, y_t),
                          batch_size=args.batch_size, shuffle=shuffle)

    train_loader = to_loader(X_train_n, y_train, shuffle=True)
    val_loader   = to_loader(X_val_n,   y_val)
    test_loader  = to_loader(X_test_n,  y_test)

    # 5. Модель, функция потерь, оптимизатор
    model     = StudentPerformanceNet(dropout_rate=args.dropout).to(device)
    criterion = nn.MSELoss()   # среднеквадратичная ошибка
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Планировщик: уменьшает LR если val_loss не улучшается 15 эпох
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    print(f"\nНачало обучения ({args.epochs} эпох, lr={args.lr}, batch={args.batch_size})")
    print("-" * 60)

    best_val_loss = float("inf")
    best_state    = None

    # 6. Цикл обучения
    for epoch in range(1, args.epochs + 1):
        train_loss          = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae   = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # Сохраняет лучшую модель по валидационной ошибке
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        # Печатает каждые 10 эпох
        if epoch % 10 == 0 or epoch == 1:
            print(f"Эпоха {epoch:4d}/{args.epochs} | "
                  f"Train MSE: {train_loss:8.2f} | "
                  f"Val MSE: {val_loss:8.2f} | "
                  f"Val MAE: {val_mae:5.2f}")

    print("-" * 60)

    # 7. Финальная оценка на тестовой выборке
    model.load_state_dict(best_state)
    test_loss, test_mae = evaluate(model, test_loader, criterion, device)
    print(f"\nРезультат на тестовой выборке:")
    print(f"   MSE : {test_loss:.2f}")
    print(f"   MAE : {test_mae:.2f}  (средняя абсолютная ошибка в баллах)")
    print(f"   RMSE: {test_loss**0.5:.2f}")

    # 8. Сохранение модели
    torch.save({
        "model_state_dict": best_state,
        "dropout_rate":     args.dropout,
        "feature_names":    FEATURE_NAMES,
    }, args.model_out)
    print(f"\n[OK] Модель сохранена в '{args.model_out}'")


if __name__ == "__main__":
    main()
