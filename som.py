import numpy as np
import matplotlib.pyplot as plt


class SOM:
    def __init__(self):
        pass

    def euc_dist(self, x, y):
        sum = 0
        for i in range(len(x)):
            sum = sum + pow((x[i] - y[i]), 2)
        res = np.sqrt(sum)
        return res

    def man_dist(self, n1_x, n1_y, n2_x, n2_y):
        return abs(n1_x - n2_x) + abs(n1_y - n2_y)

    def make_node(self, data, t, map):
        row = len(map)
        col = len(map[0])
        vec = data[t]
        min_val = 1.79E+308
        x = 0
        y = 0
        for i in range(row):
            for j in range(col):
                dist = self.euc_dist(vec, map[i][j])
                if dist < min_val:
                    min_val = dist
                    x = i
                    y = j
        return x, y

    # Creating the data
    def som(self, data: np.array, max_iter: int, learning_rate: float, row: int, col: int, features: int,
            show_prog=False, desc="half iter", limit=True):
        if max_iter > len(data):
            print("The data size is smaller then the number of iterations")
            return
        range_max = row + col
        map = np.random.rand(row, col, features)
        for iter in range(max_iter):
            bmu_row, bmu_col = self.make_node(data, iter, map)

            pct_left = 1.0 - ((iter * 1.0) / max_iter)
            curr_range = (int)(pct_left * range_max)
            curr_rate = pct_left * learning_rate

            for i in range(row):
                for j in range(col):
                    if self.man_dist(bmu_row, bmu_col, i, j) < curr_range:
                        map[i][j] = map[i][j] + curr_rate * (data[iter] - map[i][j])

            if show_prog and iter == max_iter // 2:
                if limit:
                    plt.gca().set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
                plt.title(desc)
                self.plot_map(map, data)
        return map

    def som2(self, data: np.array, max_iter: int, learning_rate: float, row: int, col: int, features: int, x, y,
            show_prog=False, desc="half iter", limit=True):
        if max_iter > len(data):
            print("The data size is smaller then the number of iterations")
            return
        range_max = row + col
        map = np.random.rand(row, col, features)
        for iter in range(max_iter):
            bmu_row, bmu_col = self.make_node(data, iter, map)

            pct_left = 1.0 - ((iter * 1.0) / max_iter)
            curr_range = (int)(pct_left * range_max)
            curr_rate = pct_left * learning_rate

            for i in range(row):
                for j in range(col):
                    if self.man_dist(bmu_row, bmu_col, i, j) < curr_range:
                        map[i][j] = map[i][j] + curr_rate * (data[iter] - map[i][j])

            if show_prog and iter == max_iter // 2:
                if limit:
                    plt.gca().set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
                plt.title(desc)
                self.plot_map2(map, data, x, y)
        return map

    # Plotting the data
    def plot_map(self, map, data):
        x = []
        y = []
        for cell in data:
            x.append(cell[0])
            y.append(cell[1])
        plt.scatter(x, y)
        x = []
        y = []
        for row in range(len(map)):
            for col in range(len(map[0])):
                x.append(map[row][col][0])
                y.append(map[row][col][1])
        plt.scatter(x, y, color='red')
        plt.show()
        return x, y
    def plot_map2(self, map, data, p_x, p_y):
        x = []
        y = []
        for cell in data:
            x.append(cell[0])
            y.append(cell[1])
        plt.scatter(x, y)
        plt.scatter(p_x, p_y, color='red')
        plt.show()