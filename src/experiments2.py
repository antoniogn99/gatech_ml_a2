import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose


# Read data
DATA_CSV_PATH = "data//c2_data.csv"
df = pd.read_csv(DATA_CSV_PATH)
x = df.drop(["churn"], axis=1).values
y = df["churn"].values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

# Normalize feature data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def plot_training_nn_with_hc():
    train_scores = []
    test_scores = []
    max_iter_list = list(range(0, 5001, 1000))
    for max_iter in max_iter_list:
        model  = mlrose.NeuralNetwork(
            hidden_nodes=[32],
            activation = 'relu',
            algorithm='random_hill_climb',
            max_iters=max_iter,
            bias=True,
            is_classifier=True,
            learning_rate = 0.1,
            early_stopping=True,
            clip_max=5,
            max_attempts=100,
            random_state=0,
            curve=True,
        )
        model.fit(X_train_scaled, y_train)
        train_scores.append(f1_score(y_train, model.predict(X_train_scaled), average="weighted"))
        test_scores.append(f1_score(y_test, model.predict(X_test_scaled), average="weighted"))
    plt.plot(max_iter_list, train_scores, "o-", label="Training")
    plt.plot(max_iter_list, test_scores, "o-", label="Test")
    plt.xlabel("Number of iterations")
    plt.ylabel("Weighted F1 Score")
    plt.legend()
    plt.title("Hill Climbing")
    plt.ylim([0, 1])
    plt.savefig("nn_training_hc.png")
    plt.clf()
    plt.plot(model.fitness_curve)
    plt.xlabel("Number of iterations")
    plt.ylabel("Fitness curve")
    plt.title("Hill Climbing")
    plt.savefig("nn_training_hc_curve.png")
    plt.clf()

def plot_training_nn_with_ga():
    train_scores = []
    test_scores = []
    max_iter_list = list(range(0, 5001, 1000))
    for max_iter in max_iter_list:
        model  = mlrose.NeuralNetwork(
            hidden_nodes=[32],
            activation = 'sigmoid',
            algorithm='genetic_alg',
            max_iters=max_iter,
            bias=True,
            is_classifier=True,
            learning_rate = 0.001,
            early_stopping=True,
            clip_max=5,
            max_attempts=100,
            random_state=0,
            curve=True,
            pop_size=30,
            mutation_prob=0.1,
        )
        model.fit(X_train_scaled, y_train)
        train_scores.append(f1_score(y_train, model.predict(X_train_scaled), average="weighted"))
        test_scores.append(f1_score(y_test, model.predict(X_test_scaled), average="weighted"))
    plt.plot(max_iter_list, train_scores, "o-", label="Training")
    plt.plot(max_iter_list, test_scores, "o-", label="Test")
    plt.xlabel("Number of iterations")
    plt.ylabel("Weighted F1 Score")
    plt.legend()
    plt.title("Genetic Algorithm")
    plt.ylim([0, 0.8])
    plt.savefig("nn_training_ga.png")
    plt.clf()
    plt.plot(model.fitness_curve)
    plt.xlabel("Number of iterations")
    plt.ylabel("Fitness curve")
    plt.title("Genetic Algorithm")
    plt.savefig("nn_training_ga_curve.png")
    plt.clf()

def plot_training_nn_with_sa():
    train_scores = []
    test_scores = []
    max_iter_list = list(range(0, 5001, 1000))
    for max_iter in max_iter_list:
        model  = mlrose.NeuralNetwork(
            hidden_nodes=[32],
            activation = 'sigmoid',
            algorithm='simulated_annealing',
            max_iters=max_iter,
            bias=True,
            is_classifier=True,
            learning_rate = 0.1,
            early_stopping=True,
            clip_max=5,
            max_attempts=100,
            random_state=0,
            curve=True,
        )
        model.fit(X_train_scaled, y_train)
        train_scores.append(f1_score(y_train, model.predict(X_train_scaled), average="weighted"))
        test_scores.append(f1_score(y_test, model.predict(X_test_scaled), average="weighted"))
    plt.plot(max_iter_list, train_scores, "o-", label="Training")
    plt.plot(max_iter_list, test_scores, "o-", label="Test")
    plt.xlabel("Number of iterations")
    plt.ylabel("Weighted F1 Score")
    plt.legend()
    plt.title("Simulated Annealing")
    plt.ylim([0, 0.8])
    plt.savefig("nn_training_sa.png")
    plt.clf()
    plt.plot(model.fitness_curve)
    plt.xlabel("Number of iterations")
    plt.ylabel("Fitness curve")
    plt.title("Simulated Annealing")
    plt.savefig("nn_training_sa_curve.png")
    plt.clf()

plot_training_nn_with_ga()