import signal
import numpy as np
from matplotlib import pyplot as plt
from sympy import false
from scipy import signal, ndimage
import control

def stability(transfer_function):
    poles = transfer_function.poles()
    if np.all(np.real(poles) <= 0):
        if np.any(np.real(poles) == 0):
            print('Один из полюсов находится на мнимой оси - система на грани устойчивости')
            return False
        else:
            print('Все полюса передаточной функции левые -  система устойчива')
            return True
    else:
        print('Имеются правые полюса -  система не устойчива')
        return false

def Perehod_character(res_function):

    [x, y] = control.step_response(res_function)
    plt.plot(x, y)
    plt.xlabel('Time, с')
    plt.ylabel('Amplitude')
    plt.title('Переходная характеристика h(t)')
    plt.grid(True)
    plt.show()


def Transition_time (transfer_function):
    # Генерация временной оси для переходного процесса
    t, y = control.step_response(transfer_function)

    # Определение времени, когда значение выходит из диапазона от 0,95 до 1,05
    end_time = None  # конечное время
    start_time = None  # начальное время

    for i in range(len(y) - 1, -1, -1):
        if 0.95*y[len(y) - 1] <= y[i] <= 1.05*y[len(y) - 1]:
            end_time = t[i]
        else:
            start_time = t[i + 1]
            break

    if end_time is not None and start_time is not None:
        print("Длительность переходного процесса ", end_time, 'c')
        # Визуализация переходной характеристики с прямой, проходящей через найденное время
        plt.plot(t, y, label='Переходная характеристика')
        plt.axvline(x=start_time, color='r', linestyle='--', label='Начало диапазона')
        plt.axvline(x=end_time, color='g', linestyle='--', label='Выход из диапазона')
        plt.axhline(y=0.95 * y[len(y) - 1], color='b', linestyle='--', label='0.95 * y[len(y) - 1]')
        plt.axhline(y=1.05 * y[len(y) - 1], color='m', linestyle='--', label='1.05 * y[len(y) - 1]')
        plt.legend()
        plt.xlabel('Время')
        plt.ylabel('Значение')
        plt.title('Переходная характеристика с прямой')
        plt.grid(True)
        plt.show()
        return end_time
    else:
        print("Заданный диапазон значений не был достигнут.")
        return None

def Over_regulation(transfer_function):
    t, y = control.step_response(transfer_function)
    extrema_indices = signal.argrelextrema(y, np.greater)

    if len(extrema_indices[0]) == 0:
        print("Не найдено экстремумов в переходной характеристике.")
        return None

    max_index = np.argmax(y[extrema_indices])
    max_time = t[extrema_indices][max_index]
    max_value = y[extrema_indices][max_index]

    if max_value <= y[len(y) - 1]:
        print("Наибольший экстремум не превышает установившееся значение")
        return None

    plt.plot(t, y, label='Переходная характеристика')
    plt.plot(max_time, max_value, 'ro', label='Наибольший экстремум')
    plt.legend()
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.title('Переходная характеристика с наибольшим экстремумом')
    plt.grid(True)
    plt.show()
    print('Перерегулирования', 100*(max_value-y[len(y)-1])/(y[len(y)-1]),'%')
    return 100*(max_value-y[len(y)-1])/(y[len(y)-1])


def Hesitation(transfer_function, end_time):
    t, y = control.step_response(transfer_function)
    extrema_indices = signal.argrelextrema(y, np.greater)
    if len(extrema_indices[0]) == 0:
        print("Колебательность отсутствует")
        return None

    max_index = np.argmax(y[extrema_indices])
    max_time = t[extrema_indices][max_index]
    max_value = y[extrema_indices][max_index]

    if max_value <= y[len(y) - 1]:
        print("Колебательность отсутствует")
        return None

    if len(extrema_indices[0]) >= 2:
        second_max_index = np.argmax(y[extrema_indices][1:])
        second_max_time = t[extrema_indices][1:][second_max_index]
        first_max_time = max_time

        print('Колебательность = ', (end_time)/(second_max_time - first_max_time))
        return (end_time)/(second_max_time - first_max_time)


# Метод АДАМА
# Целевая функция в виде перерегулирования
def loss_function(transfer_function):
    t, y = control.step_response(transfer_function)
    extrema_indices = signal.argrelextrema(y, np.greater)

    if len(extrema_indices[0]) == 0:
        return np.inf

    max_index = np.argmax(y[extrema_indices])
    max_value = y[extrema_indices][max_index]

    if max_value <= y[len(y) - 1]:
        return np.inf

    return 100*(max_value-y[len(y)-1])/(y[len(y)-1])


def gradient_loss(closed_loop_tf, epsilon=1e-6):
    numerator = closed_loop_tf.num[0][0]
    denominator = closed_loop_tf.den[0][0]
    closed_loop_tf_array = np.concatenate((numerator, denominator))
    grad = np.zeros_like(closed_loop_tf_array)

    for i in range(len(closed_loop_tf_array)):
        perturbed_tf = control.TransferFunction(numerator, denominator)
        perturbed_tf.num[0][0][i] += epsilon

        loss_perturbed = loss_function(perturbed_tf)
        loss_original = loss_function(closed_loop_tf)

        grad[i] = (loss_perturbed - loss_original) / epsilon

    return grad




Kp = 1
Ki = 5
Kd = 1
numerator = [1.19338538, 5.46212394]   # числитель
denominator = [1.09904975, 7.89119726, 13.24697323, 10.92432842] # знаменатель
w_ooc = control.TransferFunction(numerator, denominator) # Передаточная функция объекта управления
pid_tf = control.TransferFunction([Kd, Kp, Ki], [1, 0])# Передаточная функция ПИД-регулятора
closed_loop_tf = control.feedback(pid_tf*w_ooc)# Передаточная замкнутой системы с отрицательной обратной связью


if stability(closed_loop_tf):
    Perehod_character(closed_loop_tf)
    Over_regulation(closed_loop_tf)
    Hesitation(closed_loop_tf, Transition_time(closed_loop_tf))
    print(gradient_loss(closed_loop_tf))







