import numpy as np

# Setting the seed
np.random.seed(55)


# Fitness function
def fitness(x):
    return x ** 2


# Initial solution
current_solution = np.random.randn() * 10

# Tabu list
tabu_list = []
tabu_length = 100  # Assuming a tabu length of 100, you can adjust this value
best_solution = current_solution
best_fitness = fitness(current_solution)

while best_fitness > 1e-10:

    # Generate neighbors
    neighbors = current_solution + np.random.randn(50)

    # Filter out neighbors in the tabu list
    feasible_neighbors = [n for n in neighbors if n not in tabu_list]

    # Evaluate fitness
    neighbors_fitness = [fitness(n) for n in feasible_neighbors]

    # If no neighbors are feasible, break out
    if not feasible_neighbors:
        break

    # Get the best neighbor
    best_neighbor_idx = np.argmin(neighbors_fitness)
    best_neighbor = feasible_neighbors[best_neighbor_idx]

    # If the neighbor's fitness is better, update the current solution
    if neighbors_fitness[best_neighbor_idx] < best_fitness:
        best_solution = best_neighbor
        best_fitness = neighbors_fitness[best_neighbor_idx]

    # Update the current solution to the best neighbor
    current_solution = best_neighbor

    # Add the solution to the tabu list
    tabu_list.append(current_solution)

    # If tabu list exceeds specified length, remove the oldest entry
    if len(tabu_list) > tabu_length:
        tabu_list.pop(0)

    print(f"Best solution: {best_solution}, Best fitness: {best_fitness}")

print("Final solution:", best_solution)
