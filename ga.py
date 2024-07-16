import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from deap import base, creator, tools, algorithms

N_GENERATIONS = 20
N_POPULATION = 50

# Load dataset
data = pd.read_csv('trimmed_data.csv')

# Preprocessing
data['date'] = pd.to_datetime(data['date'])
data = data.drop(columns=['date'])
X = data.drop(columns=['score'])
y = data['score']
y = y.fillna(0.0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Genetic Algorithm Setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.randint, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(X.columns))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    selected_features = [i for i, bit in enumerate(individual) if bit == 1]
    if len(selected_features) == 0:
        return float('inf'),  # Return a high value if no features are selected

    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]

    model = XGBRegressor()
    model.fit(X_train_selected, y_train)
    predictions = model.predict(X_test_selected)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return rmse,

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def main():
    population = toolbox.population(n=N_POPULATION)
    n_generations = N_GENERATIONS
    cxpb, mutpb = 0.5, 0.2

    for gen in range(n_generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
        fits = map(toolbox.evaluate, offspring)

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))

    best_individual = tools.selBest(population, k=1)[0]
    selected_features = [X.columns[i] for i, bit in enumerate(best_individual) if bit == 1]
    print("Best individual is: ", best_individual)
    print("With features: ", selected_features)
    print("With RMSE: ", evaluate(best_individual))

if __name__ == "__main__":
    main()
