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
        # print("Длительность переходного процесса ", end_time, 'c')
        # # Визуализация переходной характеристики с прямой, проходящей через найденное время
        # plt.plot(t, y, label='Переходная характеристика')
        # plt.axvline(x=start_time, color='r', linestyle='--', label='Начало диапазона')
        # plt.axvline(x=end_time, color='g', linestyle='--', label='Выход из диапазона')
        # plt.axhline(y=0.95 * y[len(y) - 1], color='b', linestyle='--', label='0.95 * y[len(y) - 1]')
        # plt.axhline(y=1.05 * y[len(y) - 1], color='m', linestyle='--', label='1.05 * y[len(y) - 1]')
        # plt.legend()
        # plt.xlabel('Время')
        # plt.ylabel('Значение')
        # plt.title('Переходная характеристика с прямой')
        # plt.grid(True)
        # plt.show()
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
def loss_function_over_regulation(transfer_function, TZ_over_regulation ,t , y, extrema_indices):
    # t, y = control.step_response(transfer_function)
    # extrema_indices = signal.argrelextrema(y, np.greater)

    if len(extrema_indices[0]) == 0:
        return 0

    max_index = np.argmax(y[extrema_indices])
    max_value = y[extrema_indices][max_index]

    if max_value <= y[len(y) - 1]:
        return 0

    return abs((100*(max_value-y[len(y)-1])/(y[len(y)-1]) - TZ_over_regulation)/ TZ_over_regulation) * 100

def loss_function_transition_time(transfer_function, TZ_time, t, y):

    # t, y = control.step_response(transfer_function)

    end_time = None  # конечное время
    start_time = None  # начальное время

    for i in range(len(y) - 1, -1, -1):
        if 0.95 * y[len(y) - 1] <= y[i] <= 1.05 * y[len(y) - 1]:
            end_time = t[i]
        else:
            start_time = t[i + 1]
            break

    if end_time is not None and start_time is not None:
        return abs((end_time - TZ_time)/ TZ_time) * 100
    else:
        return 0


def loss_function_hesitation(transfer_function, end_time, TZ_hesitation, t, y, extrema_indices):
    # t, y = control.step_response(transfer_function)
    # extrema_indices = signal.argrelextrema(y, np.greater)
    if len(extrema_indices[0]) == 0:
        return 0

    max_index = np.argmax(y[extrema_indices])
    max_time = t[extrema_indices][max_index]
    max_value = y[extrema_indices][max_index]

    if max_value <= y[len(y) - 1]:
        return 0
    if len(extrema_indices[0]) == 1:
        return 0

    if len(extrema_indices[0]) >= 2:
        second_max_index = np.argmax(y[extrema_indices][1:])
        if second_max_index == 0:
            return 0
        second_max_time = t[extrema_indices][1:][second_max_index]
        first_max_time = max_time

        return abs(((end_time) / (second_max_time - first_max_time)) - TZ_hesitation) / TZ_hesitation * 100

def loss_function(coef_pid):
    closed_loop_tf = Closed_loop_tf(coef_pid)
    t, y = control.step_response(closed_loop_tf)
    extrema_indices = signal.argrelextrema(y, np.greater)
    TZ_time = 15  # секунд
    TZ_over_regulation = 20  # процентов
    TZ_hesitation = 1.15

    over_regulation = loss_function_over_regulation(closed_loop_tf, TZ_over_regulation, t, y, extrema_indices)
    transition_time = loss_function_transition_time(closed_loop_tf, TZ_time, t, y)
    hesitation = loss_function_hesitation(closed_loop_tf, Transition_time(closed_loop_tf), TZ_hesitation, t, y, extrema_indices)
    # + transition_time + hesitation
    return over_regulation + transition_time + hesitation


def gradient_loss(coef_pid, eps=1e-8):

    grad_Kd = (loss_function([coef_pid[0] + eps, coef_pid[1], coef_pid[2]]) - loss_function([coef_pid[0], coef_pid[1], coef_pid[2]])) / eps
    grad_Kp = (loss_function([coef_pid[0], coef_pid[1] + eps, coef_pid[2]]) - loss_function([coef_pid[0], coef_pid[1], coef_pid[2]])) / eps
    grad_Ki = (loss_function([coef_pid[0], coef_pid[1], coef_pid[2] + eps]) - loss_function([coef_pid[0], coef_pid[1], coef_pid[2]])) / eps

    return [grad_Kd, grad_Kp, grad_Ki]

def Closed_loop_tf(coef):
    numerator = [1.19338538, 5.46212394]  # числитель
    denominator = [1.09904975, 7.89119726, 13.24697323, 10.92432842]  # знаменатель
    w_ooc = control.TransferFunction(numerator, denominator)  # Передаточная функция объекта управления
    pid_tf = control.TransferFunction(coef, [1, 0])  # Передаточная функция ПИД-регулятора
    closed_loop_tf = control.feedback(pid_tf * w_ooc)  # Передаточная замкнутой системы с отрицательной обратной связью
    return closed_loop_tf

Kp = 1
Ki = 5
Kd = 1
# numerator = [1.19338538, 5.46212394]   # числитель
# denominator = [1.09904975, 7.89119726, 13.24697323, 10.92432842] # знаменатель
# w_ooc = control.TransferFunction(numerator, denominator) # Передаточная функция объекта управления
# pid_tf = control.TransferFunction([Kd, Kp, Ki], [1, 0])# Передаточная функция ПИД-регулятора
# closed_loop_tf = control.feedback(pid_tf*w_ooc)# Передаточная замкнутой системы с отрицательной обратной связью
coef = [Kd, Kp, Ki]

# coef = [4.194772146105173, 0.36575341586941057, 4.825805393139875]
# print(Hesitation((Closed_loop_tf(coef)), Transition_time(Closed_loop_tf(coef))))
# Perehod_character(Closed_loop_tf(coef))

#[2.072606819359821, 3.5306196215044343, 2.6050766604927804]
# coef = [2.072606819359821, 3.5306196215044343, 2.6050766604927804]
# Perehod_character(Closed_loop_tf(coef))
#
# coef = [2.0668677573832466, 3.6665790810041763, 2.7876933790728886]
# Perehod_character(Closed_loop_tf(coef))
#
# coef = [1.5633398657334368, 2.133452006082576, 1.630410668102446]
# Perehod_character(Closed_loop_tf(coef))
#
# coef = [1.998792017677677, 3.1302069151695298, 2.219334483675464]
# Perehod_character(Closed_loop_tf(coef))


def adam_optimizer(initial_parameters, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10000):
    parameters = initial_parameters
    m = np.zeros_like(parameters)
    v = np.zeros_like(parameters, dtype=np.float64)
    print(parameters)
    for t in range(1, num_iterations + 1):
        if t % 10 == 0:
            print(t)
        gradients = gradient_loss(parameters)
        j = 0
        for j in range(len(parameters)):
            m[j] = beta1 * m[j] + (1 - beta1) * gradients[j]
            v[j] = beta2 * v[j] + (1 - beta2) * (gradients[j] ** 2)

            m_hat = m[j] / (1 - beta1 ** t)
            v_hat = v[j] / (1 - beta2 ** t)
            parameters[j] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return parameters

Perehod_character(Closed_loop_tf(coef))
new_coef = adam_optimizer(coef)
print(new_coef)
Perehod_character(Closed_loop_tf(new_coef))
