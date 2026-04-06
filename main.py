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


# ============================================================
# ========================== ТЕСТЫ ===========================
# ============================================================

def test_create_vector():
    v = create_vector()
    assert isinstance(v, np.ndarray)
    assert v.shape == (10,)
    assert np.array_equal(v, np.arange(10))


def test_create_matrix():
    m = create_matrix()
    assert isinstance(m, np.ndarray)
    assert m.shape == (5, 5)
    assert np.all((m >= 0) & (m < 1))


def test_reshape_vector():
    v = np.arange(10)
    reshaped = reshape_vector(v)
    assert reshaped.shape == (2, 5)
    assert reshaped[0, 0] == 0
    assert reshaped[1, 4] == 9


def test_vector_add():
    assert np.array_equal(
        vector_add(np.array([1,2,3]), np.array([4,5,6])),
        np.array([5,7,9])
    )
    assert np.array_equal(
        vector_add(np.array([0,1]), np.array([1,1])),
        np.array([1,2])
    )


def test_scalar_multiply():
    assert np.array_equal(
        scalar_multiply(np.array([1,2,3]), 2),
        np.array([2,4,6])
    )


def test_elementwise_multiply():
    assert np.array_equal(
        elementwise_multiply(np.array([1,2,3]), np.array([4,5,6])),
        np.array([4,10,18])
    )


def test_dot_product():
    assert dot_product(np.array([1,2,3]), np.array([4,5,6])) == 32
    assert dot_product(np.array([2,0]), np.array([3,5])) == 6


def test_matrix_multiply():
    A = np.array([[1,2],[3,4]])
    B = np.array([[2,0],[1,2]])
    assert np.array_equal(matrix_multiply(A,B), A @ B)


def test_matrix_determinant():
    A = np.array([[1,2],[3,4]])
    assert round(matrix_determinant(A),5) == -2.0


def test_matrix_inverse():
    A = np.array([[1,2],[3,4]])
    invA = matrix_inverse(A)
    assert np.allclose(A @ invA, np.eye(2))


def test_solve_linear_system():
    A = np.array([[2,1],[1,3]])
    b = np.array([1,2])
    x = solve_linear_system(A,b)
    assert np.allclose(A @ x, b)


def test_load_dataset():
    # Для теста создадим временный файл
    test_data = "math,physics,informatics\n78,81,90\n85,89,88"
    with open("test_data.csv", "w", encoding='utf-8') as f:
        f.write(test_data)
    try:
        data = load_dataset("test_data.csv")
        assert data.shape == (2, 3)
        assert np.array_equal(data[0],[78,81,90])
    finally:
        os.remove("test_data.csv")


def test_statistical_analysis():
    data = np.array([10,20,30])
    result = statistical_analysis(data)
    assert result["mean"] == 20
    assert result["min"] == 10
    assert result["max"] == 30


def test_normalization():
    data = np.array([0,5,10])
    norm = normalize_data(data)
    assert np.allclose(norm, np.array([0,0.5,1]))


def test_plot_histogram():
    # Просто проверяем, что функция не падает
    data = np.array([1,2,3,4,5])
    plot_histogram(data)


def test_plot_heatmap():
    matrix = np.array([[1,0.5],[0.5,1]])
    plot_heatmap(matrix)


def test_plot_line():
    x = np.array([1,2,3])
    y = np.array([4,5,6])
    plot_line(x, y)


if __name__ == "__main__":
    print("Запустите python -m pytest main.py -v для проверки лабораторной работы.")