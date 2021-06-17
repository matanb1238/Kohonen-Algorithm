import numpy as np
import matplotlib.pyplot as plt


def draw(w1, w2, iterate, radius, l_rate, is_line=True):
    fig, ax = plt.subplots()
    if is_line:
        plt.plot(w1, w2, color='gray', marker='o', linestyle='dashed',
                 linewidth=2, markersize=12)
    else:
        plt.plot(w1, w2, color='cyan', linestyle='None', marker='o', markersize=12)
    ax.set(xlabel='X',
           ylabel='Y',
           title='Kohonen algorithm (Iter: {}, Radius: {:.3f}, L_rate:{:.3f})'
           .format(iterate, radius, l_rate))
    ax.grid()
    plt.show()


def euclidean_dist(x, y, node_id):
    return np.sqrt((x[node_id] - x) ** 2 + (y[node_id] - y) ** 2)


def bmu(x, y, node_id):
    ed = euclidean_dist(x, y, node_id)
    ed[node_id] = np.inf
    argmin_x = np.argmin(ed).item()
    return argmin_x


def choose_randomly(x):
    rnd = np.where(x == np.random.choice(x))
    return rnd if type(rnd) is not tuple else int(rnd[0][0])


def ord_vector(x, y, node_id):
    by_euclid_dist = -1 * euclidean_dist(x, y, node_id)
    x_ord_idx = np.argsort(by_euclid_dist)
    y_ord_idx = np.argsort(by_euclid_dist)
    return x[x_ord_idx], y[y_ord_idx]


def update_neighbourhood(x, y, bmu_node, node_id, radius, alpha):
    ed = euclidean_dist(x, y, bmu_node)
    for idx, r in enumerate(ed):
        if -radius <= r <= radius:
            x[idx] += alpha * (x[node_id] - x[idx])
            y[idx] += alpha * (y[node_id] - y[idx])
    return x, y


def train(x, y, max_iter, radius, alpha, is_line=True):
    init_alpha = alpha
    init_radius = radius
    for step in range(max_iter):
        choice_idx = choose_randomly(x)
        best_match = bmu(x, y, choice_idx)
        x, y = update_neighbourhood(x, y, best_match, choice_idx, radius, alpha)
        x, y = ord_vector(x, y, choice_idx)
        alpha = init_alpha * np.exp(-step / max_iter)
        radius = init_radius * np.exp(-step / max_iter)
        if step == 100:
            draw(x, y, step, radius, alpha, is_line)
        elif (step + 1) % 500 == 0:
            draw(x, y, (step + 1), radius, alpha, is_line)
    return x, y