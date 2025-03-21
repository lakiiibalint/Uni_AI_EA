
import random
import matplotlib.pyplot as plt

COMPLETED_SUDOKU = 243 #fitness level of a solved sudoku puzzle
THRESHOLD = 0.9 * 243 
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.5
CELL_FILL_PROBABILITY= 0.8


#Initialise
def create_random_grid():
    grid = [[0 for _ in range(9)] for _ in range(9)]  #9x9
    for i in range(9):
        for j in range(9):
            if random.random() < CELL_FILL_PROBABILITY:
                grid[i][j] = random.randint(1,9)
    return grid 

grid = create_random_grid()  # Függvény meghívása
for row in grid: 
    print (row)

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

#Plot
def plot_fitness(generations, best_fitness, average_fitness):
    plt.plot(generations, best_fitness, label="Best Fitness")
    plt.plot(generations, average_fitness, label="Average Fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness Value")
    plt.title("Fitness Over Generations")
    plt.legend()
    plt.show()

#Main 
population = [create_random_grid() for _ in range(POPULATION_SIZE)]
best_fitness_history = []
average_fitness_history = []

for generation in range(GENERATIONS):
    print(f"Generation {generation + 1}")

    fitness_values = [fitness(grid) for grid in population]
    best_fitness = max(fitness_values)
    average_fitness = sum(fitness_values) / len(fitness_values)
    best_fitness_history.append(best_fitness)
    average_fitness_history.append(average_fitness)

    if best_fitness >= THRESHOLD:
        print(f"Valid solution with fitness {best_fitness}")
        for grid in population:
            if fitness(grid) == COMPLETED_SUDOKU:
                print("Solution:")
                for row in grid:
                    print(row)
        break

    selected = selection(population)
    best = max(population, key=fitness)
    print("Selected Grids (Top 50%):")
    for i, grid in enumerate(selected):
        print(f"Grid {i+1}: {grid}, Fitness: {fitness(grid)}")


    offspring = []
    for i in range (0,len(selected),2):
        parent1= selected[i]
        parent2 = selected[i+1] if i+1 < len(selected) else selected[0]

        child1 = recombine(parent1,parent2)
        child2 = recombine(parent2,parent1)

        offspring.append(child1)
        offspring.append(child2)

    print("Offspring:")
    for i, child in enumerate (offspring):
        print(f"Child {i+1}: {child}, Fitness: {fitness(child)}")

    mutated_offspring = [mutate(child) for child in offspring]
    print("Mutated Offspring:")
    for i, child in enumerate(mutated_offspring):
        print(f"Child {i+1}: {child}, Fitness: {fitness(child)}")

    next_gen = selected + mutated_offspring
    if best not in next_gen:
        next_gen.append(best)

    sorted_next_gen = sorted(next_gen, key=lambda x: fitness(x), reverse=True)[:POPULATION_SIZE]

    print("Next Generation:")
    for i, grid in enumerate(sorted_next_gen):
        print(f"NextGen {i+1}: {grid}, Fitness: {fitness(grid)}")

    population = sorted_next_gen

if generation == GENERATIONS - 1:
    print("No valid solution found")

plot_fitness(range(1, GENERATIONS + 1), best_fitness_history, average_fitness_history)


