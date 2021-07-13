import numpy as np
import matplotlib.pyplot as plt
import Kohonen
from som import SOM


def isInHand(x, y):  # 4 fingers
    if y < 0.4:
        return True
    elif 0 < x < 0.25 and y < 0.7:
        return True
    elif 0.25 < x < 0.5 and y < 0.8:
        return True
    elif 0.5 < x < 0.75 and y < 0.9:
        return True
    elif 0.75 < x < 1 and y < 0.8:
        return True
    return False


def get_data(down, up, size, steps, y_down, y_up, x_left, x_right):
    data = []
    i = 0
    while i < size:
        data_ar = np.array(np.random.uniform(down, up, 2))
        normal_x = (int)(data_ar[0] * steps) - steps
        normal_x = size - 1 if normal_x >= size else normal_x
        if x_left[0] <= data_ar[0] <= x_right[0] and y_down[0] <= data_ar[1] <= y_up[normal_x]:
            data.append([data_ar[0], data_ar[1]])
            i += 1
    return data


def full_hand(SIZE, x_limit, y_limit, STEP):
    som = SOM()
    height = 4
    y_up_x = np.zeros(SIZE)
    x_up_y = np.zeros(SIZE)
    x_left = np.zeros(SIZE)
    x_right = np.zeros(SIZE)
    # x = np.linspace(1, 9, SIZE)
    x_up_y = np.ones(SIZE)

    y = np.linspace(1, 5, SIZE)
    x_left = np.ones(SIZE)

    y = np.linspace(1, 5, SIZE)
    x_right = 8 * np.ones(SIZE)

    # x = np.linspace(1, 2, STEP)
    # y_up_x[0: STEP] = 7 * x - 2
    y_up_x[0: STEP] = 12

    x = np.linspace(2, 3, STEP)
    y_up_x[STEP: 2 * STEP] = height

    # x = np.linspace(3, 4, STEP)
    # y_up_x[2 * STEP: 3 * STEP] = 5 * x - 7
    y_up_x[2 * STEP: 3 * STEP] = 12

    x = np.linspace(4, 5, STEP)
    y_up_x[3 * STEP: 4 * STEP] = height

    # x = np.linspace(5, 6, STEP)
    # y_up_x[4 * STEP: 5 * STEP] = 4 * x - 12
    y_up_x[4 * STEP: 5 * STEP] = 12

    x = np.linspace(6, 7, STEP)
    y_up_x[5 * STEP: 6 * STEP] = height

    # x = np.linspace(7, 8, STEP)
    # y_up_x[6 * STEP: 7 * STEP] = 2 * x - 3
    y_up_x[6 * STEP: 7 * STEP] = 12

    # x = np.linspace(8, 9, SIZE - 7 * STEP)
    # y_up_x[7 * STEP:] = -5 * x + 50

    plt.plot(x_limit, y_up_x, 'r')
    plt.plot(x_limit, x_up_y, 'r')
    plt.plot(x_left, y_limit, 'r')
    plt.plot(x_right, y_limit-y_limit, 'r')

    data = get_data(1, 13, 2000, STEP, x_up_y, y_up_x, x_left,
                    x_right)

    map = som.som(data, 2000, 0.5, 5, 6, 2, show_prog=True, desc="Neurons = 225 , 250 per Iter", limit=False)

    plt.title("Neurons = 225 , 500 per Iter")
    plt.plot(x_limit, y_up_x, 'g')
    plt.plot(x_limit, x_up_y, 'g')
    plt.plot(x_left, y_limit, 'g')
    plt.plot(x_right, y_limit, 'g')
    x, y = som.plot_map(map, data)
    cut_finger(SIZE, x_limit, y_limit, STEP, x, y)


def cut_finger(SIZE, x_limit, y_limit, STEP, p_x, p_y):
    som = SOM()
    height = 4
    y_up_x = np.zeros(SIZE)
    x_up_y = np.zeros(SIZE)
    x_left = np.zeros(SIZE)
    x_right = np.zeros(SIZE)
    x = np.linspace(1, 9, SIZE)
    x_up_y = np.ones(SIZE)
    y = np.linspace(1, 5, SIZE)
    x_left = np.ones(SIZE)
    y = np.linspace(1, 5, SIZE)
    x_right = 8 * np.ones(SIZE)

    y_up_x[0: STEP] = 12
    # x = np.linspace(3, 5, 2 * STEP)
    # y_up_x[2 * STEP: 4 * STEP] = height

    y_up_x[STEP: 3 * STEP] = height

    # x = np.linspace(3, 4, STEP)
    # y_up_x[2 * STEP: 3 * STEP] = 5 * x - 7
    # y_up_x[2 * STEP: 3 * STEP] = 12

    x = np.linspace(4, 5, STEP)
    y_up_x[3 * STEP: 4 * STEP] = height

    # x = np.linspace(5, 6, STEP)
    # y_up_x[4 * STEP: 5 * STEP] = 4 * x - 12
    y_up_x[4 * STEP: 5 * STEP] = 12

    x = np.linspace(6, 7, STEP)
    y_up_x[5 * STEP: 6 * STEP] = height

    # x = np.linspace(7, 8, STEP)
    # y_up_x[6 * STEP: 7 * STEP] = 2 * x - 3
    y_up_x[6 * STEP: 7 * STEP] = 12

    # x = np.linspace(5, 6, STEP)
    # y_up_x[4 * STEP: 5 * STEP] = 4 * x - 12
    #
    # x = np.linspace(6, 7, STEP)
    # y_up_x[5 * STEP: 6 * STEP] = height
    #
    # x = np.linspace(7, 8, STEP)
    # y_up_x[6 * STEP: 7 * STEP] = 2 * x - 6
    #
    # x = np.linspace(8, 9, SIZE - 7 * STEP)
    # y_up_x[7 * STEP:] = -5 * x + 50

    plt.plot(x_limit, y_up_x, 'r')
    plt.plot(x_limit, x_up_y, 'r')
    plt.plot(x_left, y_limit, 'r')
    plt.plot(x_right, y_limit, 'r')

    data = get_data(1, 13, 2000, STEP, x_up_y, y_up_x, x_left, x_right)
    # data.append(prev_data[0])
    # data.append(prev_data[1])
    map = som.som2(data, 2000, 0.5, 5, 6, 2, p_x, p_y, show_prog=True, desc="Neurons = 255 , 250 per Iter", limit=False)

    plt.title("Neurons = 255 , 500 per Iter")
    plt.plot(x_limit, y_up_x, 'g')
    plt.plot(x_limit, x_up_y, 'g')
    plt.plot(x_left, y_limit, 'g')
    plt.plot(x_right, y_limit, 'g')
    som.plot_map2(map, data, p_x, p_y)


def isIn3Hand(x, y):  # 3 fingers
    if y < 0.4:
        return True
    # elif 0 < x < 0.25 and y < 0.7:
    #    return True
    elif 0.25 < x < 0.5 and y < 0.8:
        return True
    elif 0.5 < x < 0.75 and y < 0.9:
        return True
    elif 0.75 < x < 1 and y < 0.8:
        return True
    return False


def sample1():
    size = 1000
    x = np.random.uniform(0, 1, size)
    y = np.random.uniform(0, 1, size)
    for i in range(size):
        if not isInHand(x[i], y[i]):
            while not isInHand(x[i], y[i]):
                x[i] = np.random.uniform(0, 1)
                y[i] = np.random.uniform(0, 1)
    step_max = 3000
    radius = 10
    alpha = 0.3
    Kohonen.train(x, y, step_max, radius, alpha)
    Kohonen.draw(x, y, 1, radius, alpha, is_line=False)


def sample2():
    size = 1000
    x = np.random.uniform(0, 1, size)
    y = np.random.uniform(0, 1, size)
    for i in range(size):
        if not isIn3Hand(x[i], y[i]):
            while not isIn3Hand(x[i], y[i]):
                x[i] = np.random.uniform(0, 1)
                y[i] = np.random.uniform(0, 1)
    step_max = 3000
    radius = 10
    alpha = 0.3
    Kohonen.train(x, y, step_max, radius, alpha)
    Kohonen.draw(x, y, 1, radius, alpha, is_line=False)


def sample3():
    min_x = 1
    max_x = 8
    SIZE = 1000
    x_limit = np.linspace(min_x, max_x, SIZE)
    y_limit = np.linspace(min_x, 12, SIZE)
    STEP = 1000 // (max_x - min_x)
    full_hand(SIZE, x_limit, y_limit, STEP)
    # cut_finger(SIZE, x_limit, y_limit, STEP)


if __name__ == '__main__':
    chosen = input("Choose a sample: ")
    if chosen == '1':
        sample1()
    elif chosen == '2':
        sample2()
    elif chosen == '3':
        sample3()