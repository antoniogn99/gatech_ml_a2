import matplotlib.pyplot as plt
import numpy as np


def f1(p):
    distance = (p[0]-100)**2 + (p[1]-100)**2
    return np.exp(-distance/1000)

def f2(p):
    a = [25, 75]
    b = [75, 25]
    distance_a = (p[0]-a[0])**2 + (p[1]-a[1])**2
    distance_b = (p[0]-b[0])**2 + (p[1]-b[1])**2
    return np.exp(-distance_a/500) + np.exp(-distance_b/100)*0.5

def f3(p):
    centers = [(10*(i+1), 10*(j+1)) for i in range(9) for j in range(9)]
    distances = [(p[0]-c[0])**2 + (p[1]-c[1])**2 for c in centers]
    distance_to_max = distances[40]
    return sum(np.exp(-d/10)*0.5 for d in distances) + np.exp(-distance_to_max/50)*1.5

def plot_function(f):
    points = [(i, j) for i in range(100) for j in range(100)]
    values = [f(p) for p in points]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    plt.scatter(xs, ys, c=values, cmap="plasma")
    plt.colorbar()
    plt.xlim([0, 100])
    plt.ylim([0, 100])

def generate_function_plots():
    plot_function(f1)
    plt.title(r"Function $f_1$")
    plt.show()
    plot_function(f2)
    plt.title(r"Function $f_2$")
    plt.show()
    plot_function(f3)
    plt.title(r"Function $f_3$")
    plt.show()

def is_in_domain(p):
    if p[0] < 0 or p[1] < 0 or p[0] > 99 or p[1] > 99:
        return False
    else:
        return True
    
def get_neighbors(p):
    i_list = [-1, 0, 1]
    j_list = [-1, 0, 1]
    neighbors = [(p[0]+i, p[1]+j) for i in i_list for j in j_list if (i, j) != (0, 0)]
    return [p for p in neighbors if is_in_domain(p)]

def optimize_with_hc(f, p):
    found_points = [p]
    for _ in range(100):
        neighbors = get_neighbors(p)
        scores = [f(x) for x in neighbors]
        best_score = max(scores)
        if f(p) > best_score:
            break
        best_index = scores.index(best_score)
        p = neighbors[best_index]
        found_points.append(p)
    return found_points

def plot_hc_evolution_1():
    plot_function(f1)
    found_points = optimize_with_hc(f1, (20, 60))
    found_points = found_points[4::5]
    print(found_points)
    xs = np.array([p[0] for p in found_points])
    ys = np.array([p[1] for p in found_points])
    plt.plot(xs, ys, color="black", linewidth=1)
    for i in range(0, len(xs)-1):
        plt.arrow(xs[i], ys[i], dx=(xs[i+1]-xs[i])/2, dy=(ys[i+1]-ys[i])/2, head_width=1, color="black")
    plt.show()

def plot_hc_evolution_2():
    plot_function(f2)
    found_points = optimize_with_hc(f2, (10, 10))
    found_points = found_points[::5]
    xs = np.array([p[0] for p in found_points])
    ys = np.array([p[1] for p in found_points])
    plt.plot(xs, ys, color="black", linewidth=1)
    for i in range(0, len(xs)-1):
        plt.arrow(xs[i], ys[i], dx=(xs[i+1]-xs[i])/2, dy=(ys[i+1]-ys[i])/2, head_width=1, color="black")
    found_points = optimize_with_hc(f2, (90, 50))
    found_points = found_points[::5]
    xs = np.array([p[0] for p in found_points])
    ys = np.array([p[1] for p in found_points])
    plt.plot(xs, ys, color="black", linewidth=1)
    for i in range(0, len(xs)-1):
        plt.arrow(xs[i], ys[i], dx=(xs[i+1]-xs[i])/2, dy=(ys[i+1]-ys[i])/2, head_width=1, color="black")
    plt.show()

def plot_hc_evolution_3():
    starting_points = [(64, 52), (6, 93), (74, 58), (56, 11), (60, 97), (25, 35)]
    plot_function(f3)
    for p in starting_points:
        found_points = optimize_with_hc(f3, p)
        xs = np.array([p[0] for p in found_points])
        ys = np.array([p[1] for p in found_points])
        plt.plot(xs, ys, color="black", linewidth=0.2)
        for i in range(0, len(xs)-1):
            plt.arrow(xs[i], ys[i], dx=(xs[i+1]-xs[i])/2, dy=(ys[i+1]-ys[i])/2, head_width=0.4, color="black")
    plt.show()


plot_hc_evolution_3()
