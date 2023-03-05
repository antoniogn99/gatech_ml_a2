import matplotlib.pyplot as plt
import numpy as np
import random


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

def get_temperature(iteration, max_iteration, min_temperature, max_temperature, num_iterations_per_cycle):
    T = (1 + np.cos(iteration * 2*np.pi / num_iterations_per_cycle))/2 * (max_temperature - min_temperature)
    return T * (1-iteration / max_iteration) + min_temperature

def plot_temperature_function(max_iteration, min_temperature, max_temperature, num_iterations_per_cycle):
    xs = list(range(max_iteration))
    ys = []
    for i in xs:
        ys.append(get_temperature(i, max_iteration, min_temperature, max_temperature, num_iterations_per_cycle))
    plt.plot(xs, ys)
    plt.show()

def optimize_with_sa(f, p, max_iteration, min_temperature, max_temperature, num_iterations_per_cycle):
    found_points = [p]
    for i in range(max_iteration):
        T = get_temperature(i, max_iteration, min_temperature, max_temperature, num_iterations_per_cycle)
        neighbors_list = get_neighbors(p)
        neighbor = random.choice(neighbors_list)
        if f(p) < f(neighbor):
            jump_probability = 1
        else:
            jump_probability = np.exp((f(neighbor) - f(p))/T)
        if random.random() < jump_probability:
            p = neighbor
        found_points.append(p)
    return found_points

def plot_sa_evolution_1():
    random.seed(0)
    starting_point = (70, 80)
    max_iteration = 1000
    min_temperature = 0.0001
    max_temperature = 0.1
    num_iterations_per_cycle = 100
    plot_function(f1)
    found_points = optimize_with_sa(f1, starting_point, max_iteration, min_temperature, max_temperature, num_iterations_per_cycle)
    found_points = found_points[::2]
    xs = np.array([p[0] for p in found_points])
    ys = np.array([p[1] for p in found_points])
    plt.plot(xs, ys, color="black", linewidth=0.3)
    plt.plot(xs[-1], ys[-1], color='white',marker='o',markerfacecolor='black',linestyle='',markersize=5, markeredgewidth=0.6)
    plt.plot(xs[0], ys[0], color='black',marker='o',markerfacecolor='white',linestyle='',markersize=5, markeredgewidth=0.6)
    plt.title(r"Simulated Annealing: Function $f_1$")
    plt.show()

def plot_sa_evolution_2():
    random.seed(0)
    starting_point = (70, 80)
    max_iteration = 1000
    min_temperature = 0.1
    max_temperature = 1
    num_iterations_per_cycle = 100
    plot_function(f2)
    found_points = optimize_with_sa(f2, starting_point, max_iteration, min_temperature, max_temperature, num_iterations_per_cycle)
    found_points = found_points[::2]
    xs = np.array([p[0] for p in found_points])
    ys = np.array([p[1] for p in found_points])
    plt.plot(xs, ys, color="black", linewidth=0.3)
    plt.plot(xs[-1], ys[-1], color='white',marker='o',markerfacecolor='black',linestyle='',markersize=5, markeredgewidth=0.6)
    plt.plot(xs[0], ys[0], color='black',marker='o',markerfacecolor='white',linestyle='',markersize=5, markeredgewidth=0.6)
    plt.title(r"Simulated Annealing: Function $f_2$")
    plt.show()

def plot_sa_evolution_3():
    random.seed(0)
    starting_point = (70, 80)
    max_iteration = 3000
    min_temperature = 0.1
    max_temperature = 1
    num_iterations_per_cycle = 100
    plot_function(f3)
    found_points = optimize_with_sa(f3, starting_point, max_iteration, min_temperature, max_temperature, num_iterations_per_cycle)
    found_points = found_points[::2]
    xs = np.array([p[0] for p in found_points])
    ys = np.array([p[1] for p in found_points])
    plt.plot(xs, ys, color="black", linewidth=0.3)
    plt.plot(xs[-1], ys[-1], color='white',marker='o',markerfacecolor='black',linestyle='',markersize=5, markeredgewidth=0.6)
    plt.plot(xs[0], ys[0], color='black',marker='o',markerfacecolor='white',linestyle='',markersize=5, markeredgewidth=0.6)
    plt.title(r"Simulated Annealing: Function $f_3$")
    plt.show()

def crossover(parent1, parent2):
    return [(parent1[0], parent2[1]), (parent2[0], parent1[1])]

def mutation(point):
    dx = random.randint(-5, 5)
    dy = random.randint(-5, 5)
    new_point = [point[0]+dx, point[1]+dy]
    if new_point[0] < 0:
        new_point[0] = 0
    if new_point[1] < 0:
        new_point[1] = 0
    if new_point[0] > 99:
        new_point[0] = 99
    if new_point[1] > 99:
        new_point[1] = 99
    return new_point

def optimize_with_ga(f, mutation_prob, num_iterations,num_points, num_points_to_keep):
    points = [(random.randint(1, 99), random.randint(1, 99)) for _ in range(num_points)]
    best_points = []
    for _ in range(num_iterations):
        new_points = []
        scores = [f(p) for p in points]
        best_points.append(points[scores.index(max(scores))])
        points_scores = list(zip(points, scores))
        points_scores.sort(key=lambda x: x[1], reverse=True)
        points = [x[0] for x in points_scores]
        new_points += points[:num_points_to_keep]
        for _ in range((num_points-num_points_to_keep)//2):
            parent1, parent2 = random.choices(points, scores, k=2)
            child1, child2 = crossover(parent1, parent2)
            if random.random() < mutation_prob:
                child1 = mutation(child1)
            if random.random() < mutation_prob:
                child2 = mutation(child2)
            new_points += [child1, child2]
        points = new_points
    return best_points

def plot_ga_evolution_1():
    random.seed(0)
    mutation_prob = 0.1
    num_iterations = 200
    num_points = 10
    num_points_to_keep = 2
    plot_function(f1)
    found_points = optimize_with_ga(f1, mutation_prob, num_iterations, num_points, num_points_to_keep)
    xs = np.array([p[0] for p in found_points])
    ys = np.array([p[1] for p in found_points])
    plt.plot(xs, ys, color="black", linewidth=0.3)
    plt.plot(xs[-1], ys[-1], color='white',marker='o',markerfacecolor='black',linestyle='',markersize=5, markeredgewidth=0.6)
    plt.plot(xs[0], ys[0], color='black',marker='o',markerfacecolor='white',linestyle='',markersize=5, markeredgewidth=0.6)
    plt.title(r"Genetic Algorithm: Function $f_1$")
    plt.show()

def plot_ga_evolution_2():
    random.seed(0)
    mutation_prob = 0.1
    num_iterations = 200
    num_points = 10
    num_points_to_keep = 2
    plot_function(f2)
    found_points = optimize_with_ga(f2, mutation_prob, num_iterations, num_points, num_points_to_keep)
    xs = np.array([p[0] for p in found_points])
    ys = np.array([p[1] for p in found_points])
    plt.plot(xs, ys, color="black", linewidth=0.3)
    plt.plot(xs[-1], ys[-1], color='white',marker='o',markerfacecolor='black',linestyle='',markersize=5, markeredgewidth=0.6)
    plt.plot(xs[0], ys[0], color='black',marker='o',markerfacecolor='white',linestyle='',markersize=5, markeredgewidth=0.6)
    plt.title(r"Genetic Algorithm: Function $f_2$")
    plt.show()

def plot_ga_evolution_3():
    random.seed(0)
    mutation_prob = 0.1
    num_iterations = 200
    num_points = 10
    num_points_to_keep = 2
    plot_function(f3)
    found_points = optimize_with_ga(f3, mutation_prob, num_iterations, num_points, num_points_to_keep)
    xs = np.array([p[0] for p in found_points])
    ys = np.array([p[1] for p in found_points])
    plt.plot(xs, ys, color="black", linewidth=0.3)
    plt.plot(xs[-1], ys[-1], color='white',marker='o',markerfacecolor='black',linestyle='',markersize=5, markeredgewidth=0.6)
    plt.plot(xs[0], ys[0], color='black',marker='o',markerfacecolor='white',linestyle='',markersize=5, markeredgewidth=0.6)
    plt.title(r"Genetic Algorithm: Function $f_3$")
    plt.show()

def optimize_with_ga_aux(f, mutation_prob, num_iterations,num_points, num_points_to_keep):
    points = [(random.randint(1, 99), random.randint(1, 99)) for _ in range(num_points)]
    initial_generation = points
    best_points = []
    for _ in range(num_iterations):
        new_points = []
        scores = [f(p) for p in points]
        best_points.append(points[scores.index(max(scores))])
        points_scores = list(zip(points, scores))
        points_scores.sort(key=lambda x: x[1], reverse=True)
        points = [x[0] for x in points_scores]
        new_points += points[:num_points_to_keep]
        for _ in range((num_points-num_points_to_keep)//2):
            parent1, parent2 = random.choices(points, scores, k=2)
            child1, child2 = crossover(parent1, parent2)
            if random.random() < mutation_prob:
                child1 = mutation(child1)
            if random.random() < mutation_prob:
                child2 = mutation(child2)
            new_points += [child1, child2]
        points = new_points
    return initial_generation, points

def plot_ga_evolution_1_aux():
    random.seed(0)
    mutation_prob = 0.1
    num_iterations = 200
    num_points = 10
    num_points_to_keep = 2
    plot_function(f1)
    initial_generation, last_generation = optimize_with_ga_aux(f1, mutation_prob, num_iterations, num_points, num_points_to_keep)
    xs = np.array([p[0] for p in initial_generation])
    ys = np.array([p[1] for p in initial_generation])
    plt.plot(xs, ys, color='black',marker='o',markerfacecolor='white',linestyle='',markersize=5, markeredgewidth=0.6)
    xs = np.array([p[0] for p in last_generation])
    ys = np.array([p[1] for p in last_generation])
    plt.plot(xs, ys, color='white',marker='o',markerfacecolor='black',linestyle='',markersize=5, markeredgewidth=0.6)
    plt.title(r"Genetic Algorithm: Function $f_1$")
    plt.show()

def plot_ga_evolution_2_aux():
    random.seed(0)
    mutation_prob = 0.1
    num_iterations = 200
    num_points = 10
    num_points_to_keep = 2
    plot_function(f2)
    initial_generation, last_generation = optimize_with_ga_aux(f2, mutation_prob, num_iterations, num_points, num_points_to_keep)
    xs = np.array([p[0] for p in initial_generation])
    ys = np.array([p[1] for p in initial_generation])
    plt.plot(xs, ys, color='black',marker='o',markerfacecolor='white',linestyle='',markersize=5, markeredgewidth=0.6)
    xs = np.array([p[0] for p in last_generation])
    ys = np.array([p[1] for p in last_generation])
    plt.plot(xs, ys, color='white',marker='o',markerfacecolor='black',linestyle='',markersize=5, markeredgewidth=0.6)
    plt.title(r"Genetic Algorithm: Function $f_2$")
    plt.show()

def plot_ga_evolution_3_aux():
    random.seed(0)
    mutation_prob = 0.1
    num_iterations = 200
    num_points = 10
    num_points_to_keep = 2
    plot_function(f3)
    initial_generation, last_generation = optimize_with_ga_aux(f3, mutation_prob, num_iterations, num_points, num_points_to_keep)
    xs = np.array([p[0] for p in initial_generation])
    ys = np.array([p[1] for p in initial_generation])
    plt.plot(xs, ys, color='black',marker='o',markerfacecolor='white',linestyle='',markersize=5, markeredgewidth=0.6)
    xs = np.array([p[0] for p in last_generation])
    ys = np.array([p[1] for p in last_generation])
    plt.plot(xs, ys, color='white',marker='o',markerfacecolor='black',linestyle='',markersize=5, markeredgewidth=0.6)
    plt.title(r"Genetic Algorithm: Function $f_3$")
    plt.show()


class Region:
    def __init__(self, balls_centers=None, balls_radio=1):
        self.balls_centers = balls_centers
        self.balls_radio = balls_radio
    
    def contains(self, point):
        if self.balls_centers is None:
            return True
        for c in self.balls_centers:
            if (point[0]-c[0])**2 + (point[1]-c[1])**2 < self.balls_radio**2:
                return True
        return False
    
    def get_contained_points(self):
        points = [(i, j) for i in range(100) for j in range(100)]
        return [p for p in points if self.contains(p)]
    
    def generate_uniformly(self, k):
        points = self.get_contained_points()
        return random.choices(points, k=k)
    
    def plot(self, color):
        points = self.get_contained_points()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, "s", color=color)

def optimize_with_mimic(f, num_iterations, num_points, initial_balls_radio, percentil):
    regions_list = []
    thresholds_list = [0]
    region = Region()
    for i in range(num_iterations):
        balls_radio = -i*initial_balls_radio/(num_iterations-1) + initial_balls_radio + 1
        points = region.generate_uniformly(num_points)
        scores = [f(p) for p in points]
        threshold = np.percentile(scores, percentil)
        thresholds_list.append(threshold)
        best_points = [points[i] for i in range(num_points) if scores[i] > threshold]
        region = Region(best_points, balls_radio)
        regions_list.append(region)
    return regions_list, thresholds_list

def plot_mimic_evolution_1():
    random.seed(0)
    num_iterations = 50
    num_points = 40
    percentil = 50
    balls_radio = 20
    regions_list, thresholds_list = optimize_with_mimic(f1, num_iterations, num_points, balls_radio, percentil)
    points = [(i, j) for i in range(100) for j in range(100)]
    values = []
    for p in points:
        index = num_iterations-1
        while index > 0 and not regions_list[index].contains(p):
            index -= 1
        values.append(thresholds_list[index])
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    plt.scatter(xs, ys, c=values, cmap="plasma")
    plt.colorbar()
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.title(r"MIMIC: Function $f_1$")
    plt.show()


def plot_mimic_evolution_2():
    random.seed(0)
    num_iterations = 25
    num_points = 100
    percentil = 50
    balls_radio = 20
    regions_list, thresholds_list = optimize_with_mimic(f2, num_iterations, num_points, balls_radio, percentil)
    points = [(i, j) for i in range(100) for j in range(100)]
    values = []
    for p in points:
        index = num_iterations-1
        while index > 0 and not regions_list[index].contains(p):
            index -= 1
        values.append(thresholds_list[index])
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    plt.scatter(xs, ys, c=values, cmap="plasma")
    plt.colorbar()
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.title(r"MIMIC: Function $f_2$")
    plt.show()

def plot_mimic_evolution_3():
    random.seed(0)
    num_iterations = 100
    num_points = 1000
    percentil = 75
    balls_radio = 15
    regions_list, thresholds_list = optimize_with_mimic(f3, num_iterations, num_points, balls_radio, percentil)
    points = [(i, j) for i in range(100) for j in range(100)]
    values = []
    for p in points:
        index = num_iterations-1
        while index > 0 and not regions_list[index].contains(p):
            index -= 1
        values.append(thresholds_list[index])
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    plt.scatter(xs, ys, c=values, cmap="plasma")
    plt.colorbar()
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.title(r"MIMIC: Function $f_3$")
    plt.show()

plot_mimic_evolution_3()

