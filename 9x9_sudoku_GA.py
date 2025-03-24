
import random
import matplotlib.pyplot as plt
from sudoku import Sudoku
import numpy as np

COMPLETED_SUDOKU = 243 #fitness level of a solved sudoku puzzle
OPTIMAL_RESULT = 0.9 * COMPLETED_SUDOKU 
POPULATION_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.1
DIFFICULTY = 0.99
NUM_RUNS = 10


#Initialse
sudoku_generator = Sudoku(3)

def create_sudoku(DIFFICULTY):
    puzzle = Sudoku(3, seed=random.randint(0, 1000)).difficulty(DIFFICULTY)
    return puzzle.board 

#Evaluate
def fitness(grid):
    score = 0

    #Rows
    for row in grid:
        score += len(set(row)) 
    #Columns
    for col in range(9):
        column = [grid[row][col] for row in range(9)]
        score += len(set(column))
    #3x3 grids
    for i in range(0,9,3):
        for j in range(0,9,3):
            subgrid= [grid[x][y] for x in range(i, i+3) for y in range(j, j+3)]
            score += len(set(subgrid))
    return score

#Select
def selection(population):
    sorted_by_fitness = sorted(population,key= lambda x:fitness(x), reverse=True)

    return sorted_by_fitness[:POPULATION_SIZE//2]

#Recombine 
def recombine(parent1, parent2):
    child = [[0 for _ in range(9)] for _ in range(9)] 
    split = random.randint(1, 8) #crossover-point

    for i in range(split):
        child[i] = parent1[i][:]  
    for i in range(split, 9):
        child[i] = parent2[i][:]

    return child

#Mutate 
def mutate(child):
    for i in range(9):
        for j in range(9): 
            if random.random() < MUTATION_RATE:
                child[i][j] = random.randint(1,9)
    return child

#Sudoku solver
def run_sudoku_solver():
    population = [create_sudoku(DIFFICULTY) for _ in range(POPULATION_SIZE)]
    best_fitness_history = []
    average_fitness_history = []
    solution_found = False

    for generation in range(GENERATIONS):
        fitness_values = [fitness(grid) for grid in population]
        best_fitness = max(fitness_values)
        average_fitness = sum(fitness_values) / len(fitness_values)
        best_fitness_history.append(best_fitness)
        average_fitness_history.append(average_fitness)

        if best_fitness >= OPTIMAL_RESULT:
            solution_found = True

        selected = selection(population)
        best = max(population, key=fitness)

        offspring = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]

            child1 = recombine(parent1, parent2)
            child2 = recombine(parent2, parent1)

            offspring.append(child1)
            offspring.append(child2)

        mutated_offspring = [mutate(child) for child in offspring]

        next_gen = selected + mutated_offspring
        if best not in next_gen:
            next_gen.append(best)

        sorted_next_gen = sorted(next_gen, key=lambda x: fitness(x), reverse=True)[:POPULATION_SIZE]
        population = sorted_next_gen

    return best_fitness_history, average_fitness_history, solution_found


# Run it multiple times
def run_multiple_tests(NUM_RUNS):
    all_best_fitness=[]
    all_average_fitness=[]
    solutions_found= 0

    for run in range(NUM_RUNS):
        print(f"Run {run + 1}/{NUM_RUNS}")
        best_fitness_history, average_fitness_history, solution_found = run_sudoku_solver()
        all_best_fitness.append(best_fitness_history)
        all_average_fitness.append(average_fitness_history)
        if solution_found:
            solutions_found += 1

    return all_best_fitness, all_average_fitness, solutions_found
   


# Plot results 
def plot_multiple_runs(all_best_fitness, all_average_fitness, solutions_found, NUM_RUNS):
    generations = range(1, len(all_best_fitness[0]) + 1)

    avg_best_fitness = np.mean(all_best_fitness, axis=0)
    avg_average_fitness = np.mean(all_average_fitness, axis=0)
    max_best_fitness = np.max(all_best_fitness, axis=0) 

    plt.figure(figsize=(12, 7))
    
    # Plot averages
    plt.plot(generations, avg_best_fitness, label="Avg. Best Fitness", color='blue', alpha=0.7)
    plt.plot(generations, avg_average_fitness, label="Avg. Population Fitness", color='green', alpha=0.7)
    
    # Plot absolute best
    plt.plot(generations, max_best_fitness, label="Absolute Best Fitness", color='red', linestyle=':', linewidth=2)

    plt.axhline(y=OPTIMAL_RESULT, color='orange', linestyle='--', 
                linewidth=1, label=f"Optimal Threshold (218.7)")

    plt.xlabel("Generations")
    plt.ylabel("Fitness Value")
    plt.title(f"Fitness Over Generations ({NUM_RUNS} Runs)\nSolutions Found: {solutions_found}/{NUM_RUNS}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    info_text = (
        f"POPULATION_SIZE = {POPULATION_SIZE}\n"
        f"GENERATIONS = {GENERATIONS}\n"
        f"MUTATION_RATE = {MUTATION_RATE}\n"
        f"DIFFICULTY = {DIFFICULTY}\n"
        f"Solutions found = {solutions_found}/{NUM_RUNS}"
    )

    
    plt.annotate(info_text, xy=(0.98, 0.20), xycoords='axes fraction', 
                 fontsize=10, bbox=dict(boxstyle="round", alpha=0.1, facecolor='white'),
                 verticalalignment='top', horizontalalignment='right')

    plt.legend()
    plt.show()


# Main 
if __name__ == "__main__":
    all_best_fitness, all_average_fitness, solutions_found = run_multiple_tests(NUM_RUNS)
    plot_multiple_runs(all_best_fitness, all_average_fitness, solutions_found, NUM_RUNS)


