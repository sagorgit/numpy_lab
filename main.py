"""
Лабораторная работа: Численные вычисления и анализ данных с использованием NumPy.
Этот модуль содержит функции для работы с массивами, векторами, матрицами, 
а также для статистического анализа и визуализации данных.
"""

import os
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. СОЗДАНИЕ И ОБРАБОТКА МАССИВОВ
# ============================================================

def create_vector() -> np.ndarray:
    """
    Создать массив от 0 до 9.

    Returns:
        np.ndarray: Массив чисел от 0 до 9 включительно.
    """
    return np.arange(10)


def create_matrix() -> np.ndarray:
    """
    Создать матрицу 5x5 со случайными числами[0,1].

    Returns:
        np.ndarray: Матрица 5x5 со случайными значениями от 0 до 1.
    """
    return np.random.rand(5, 5)


def reshape_vector(vec: np.ndarray) -> np.ndarray:
    """
    Преобразовать (10,) -> (2,5).
    
    Args:
        vec (np.ndarray): Входной массив формы (10,).
    
    Returns:
        np.ndarray: Преобразованный массив формы (2, 5).
    """
    return vec.reshape(2, 5)


def transpose_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Транспонирование матрицы.
    
    Args:
        mat (np.ndarray): Входная матрица.
    
    Returns:
        np.ndarray: Транспонированная матрица.
    """
    return mat.T


# ============================================================
# 2. ВЕКТОРНЫЕ ОПЕРАЦИИ
# ============================================================

def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Сложение векторов одинаковой длины (векторизация без циклов).
    
    Args:
        a (np.ndarray): Первый вектор.
        b (np.ndarray): Второй вектор.
    
    Returns:
        np.ndarray: Результат поэлементного сложения.
    """
    return a + b


def scalar_multiply(vec: np.ndarray, scalar: Union[float, int]) -> np.ndarray:
    """
    Умножение вектора на число.
    
    Args:
        vec (np.ndarray): Входной вектор.
        scalar (float/int): Число для умножения.
    
    Returns:
        np.ndarray: Результат умножения вектора на скаляр.
    """
    return vec * scalar


def elementwise_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Поэлементное умножение.
    
    Args:
        a (np.ndarray): Первый вектор/матрица.
        b (np.ndarray): Второй вектор/матрица.
    
    Returns:
        np.ndarray: Результат поэлементного умножения.
    """
    return a * b


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Скалярное произведение.
    
    Args:
        a (np.ndarray): Первый вектор.
        b (np.ndarray): Второй вектор.
    
    Returns:
        float: Скалярное произведение векторов.
    """
    return float(np.dot(a, b))


# ============================================================
# 3. МАТРИЧНЫЕ ОПЕРАЦИИ
# ============================================================

def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Умножение матриц.
    
    Args:
        a (np.ndarray): Первая матрица.
        b (np.ndarray): Вторая матрица.
    
    Returns:
        np.ndarray: Результат умножения матриц.
    """
    return a @ b


def matrix_determinant(a: np.ndarray) -> float:
    """
    Определитель матрицы.
    
    Args:
        a (np.ndarray): Квадратная матрица.
    
    Returns:
        float: Определитель матрицы.
    """
    return float(np.linalg.det(a))


def matrix_inverse(a: np.ndarray) -> np.ndarray:
    """
    Обратная матрица.
    
    Args:
        a (np.ndarray): Квадратная матрица.
    
    Returns:
        np.ndarray: Обратная матрица.
    """
    return np.linalg.inv(a)


def solve_linear_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Решить систему Ax = b.
    
    Args:
        a (np.ndarray): Матрица коэффициентов A.
        b (np.ndarray): Вектор свободных членов b.
    
    Returns:
        np.ndarray: Решение системы x.
    """
    return np.linalg.solve(a, b)


# ============================================================
# 4. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================

def load_dataset(path: str = "data/students_scores.csv") -> np.ndarray:
    """
    Загрузить CSV и вернуть NumPy массив.
    
    Args:
        path (str): Путь к CSV файлу.
    
    Returns:
        np.ndarray: Загруженные данные в виде массива.
    """
    return pd.read_csv(path).to_numpy()


def statistical_analysis(data: np.ndarray) -> Dict[str, Any]:
    """
    Анализ результатов экзамена (среднее, медиана, стд, мин, макс, перцентили).
    
    Args:
        data (np.ndarray): Одномерный массив данных.
    
    Returns:
        dict: Словарь со статистическими показателями.
    """
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "25th_percentile": float(np.percentile(data, 25)),
        "75th_percentile": float(np.percentile(data, 75))
    }


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Min-Max нормализация (x - min) / (max - min).
    
    Args:
        data (np.ndarray): Входной массив данных.
    
    Returns:
        np.ndarray: Нормализованный массив данных [0, 1].
    """
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min)


# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_histogram(data: np.ndarray) -> None:
    """
    Построить гистограмму распределения оценок по математике.
    
    Args:
        data (np.ndarray): Данные для гистограммы.
    """
    os.makedirs("plots", exist_ok=True)
    plt.figure()
    plt.hist(data, bins=5, color='skyblue', edgecolor='black')
    plt.title("Распределение оценок по математике")
    plt.xlabel("Оценки")
    plt.ylabel("Количество студентов")
    plt.savefig("plots/histogram.png")
    plt.close()


def plot_heatmap(matrix: np.ndarray) -> None:
    """
    Построить тепловую карту корреляции предметов.
    
    Args:
        matrix (np.ndarray): Матрица корреляции.
    """
    os.makedirs("plots", exist_ok=True)
    plt.figure()
    sns.heatmap(matrix, annot=True, cmap="coolwarm")
    plt.title("Тепловая карта корреляции")
    plt.savefig("plots/heatmap.png")
    plt.close()


def plot_line(x: np.ndarray, y: np.ndarray) -> None:
    """
    Построить график зависимости: студент -> оценка по математике.
    
    Args:
        x (np.ndarray): Номера студентов.
        y (np.ndarray): Оценки студентов.
    """
    os.makedirs("plots", exist_ok=True)
    plt.figure()
    plt.plot(x, y, marker='o', linestyle='-', color='green')
    plt.title("Оценки студентов по математике")
    plt.xlabel("Номера студентов")
    plt.ylabel("Оценки")
    plt.savefig("plots/line_plot.png")
    plt.close()

