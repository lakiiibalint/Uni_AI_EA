#https://nidragedd.github.io/sudoku-genetics/

import random
COMPLETED_SUDOKU = 18
POPULATION_SIZE = 10
GENERATIONS = 100
MUTATION_RATE = 0.2

#Initialise
def create_random_grid():
    return [random.randint(1,3) for i in range(9)]

#Evaluate
def fitness(grid):
    score = 0

    #Rows
    for i in range(0,9,3):
        row = grid[i:i+3]
        score += len(set(row)) 
    #Columns
    for i in range(3):
        column = [grid[i], grid[i+3], grid[i+6]]
        score += len(set(column))
    
    return score

#Select
def selection(population):
    sorted_by_fitness = sorted(population,key= lambda x:fitness(x), reverse=True)

    return sorted_by_fitness[:POPULATION_SIZE//2]

#Recombine (single-point crossover)
def recombine(parent1, parent2):
    split = random.randint(1,8)

    child = parent1[:split] + parent2[split:]

    return child

#Mutate
def mutate(child):
    for i in range(len(child)):
        if random.random() < MUTATION_RATE:
            child[i] = random.randint(1,3)
    return child

population = [create_random_grid() for i in range(POPULATION_SIZE)] 

for generation in range(GENERATIONS):
    print(f"Generation {generation + 1}")

    print("Population:")
    for i, grid in enumerate(population):
        print(f"Grid {i+1}: {grid}, Fitness: {fitness(grid)}")
    
    best_fitness = max(fitness(grid) for grid in population)
    if best_fitness == COMPLETED_SUDOKU: 
        print("Valid solution found!")
        for grid in population:
            if fitness(grid) == COMPLETED_SUDOKU:
                print(f"Solution: {grid}")
        break

    selected = selection(population)
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
    sorted_next_gen = sorted(next_gen, key=lambda x: fitness(x), reverse=True)

    print("Next Generation:")
    for i, grid in enumerate(sorted_next_gen):
        print(f"NextGen {i+1}: {grid}, Fitness: {fitness(grid)}")

    population = next_gen

if generation == GENERATIONS - 1:
    print("No valid solution found")