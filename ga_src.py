import random

def read_data(filename):
    no_guests, no_tables, table_capacity = map(int, input().split())
    interests = [list(map(int, input().split())) for _ in range(no_guests)]
    guests = [input().split() for _ in range(no_guests)]

    couples = {}
    for i, guest in enumerate(guests):
        partner = int(guest[0])
        if partner != 0:
            couples[i + 1] = partner 

    return no_guests, no_tables, table_capacity, interests, guests, couples


filename = "weddingdata_20_4_50_25.dat"
no_guests, no_tables, table_capacity, interests, guests, couples = read_data(filename)

def initialize_population(no_guests, no_tables, couples, population_size):
    population = []
    for _ in range(population_size):
        individual = [0] * no_guests  
        for guest in range(1, no_guests + 1):
            if guest in couples and individual[guest - 1] == 0:
                table = random.randint(1, no_tables)
                individual[guest - 1] = table
                individual[couples[guest] - 1] = table
            elif individual[guest - 1] == 0:
                individual[guest - 1] = random.randint(1, no_tables)
        population.append(individual)
    return population


population_size = 100
population = initialize_population(no_guests, no_tables, couples, population_size)

def select_parents(population, fitness_scores, tournament_size=3):
    selected_parents = []
    for _ in range(2):  # Select two parents
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        winner = max(tournament, key=lambda item: item[1])[0]
        selected_parents.append(winner)
    return selected_parents[0], selected_parents[1]


def calculate_fitness(individual, interests, guests, table_capacity, no_tables):
    score = 0
    tables = {i: [] for i in range(1, no_tables + 1)}

    for guest_index, table in enumerate(individual):
        tables[table].append(guest_index)

    for table, guest_indices in tables.items():
        # Penalty for exceeding table capacity
        if len(guest_indices) > table_capacity:
            score -= 1000 * (len(guest_indices) - table_capacity)

        # Add score for shared interests
        for i in range(len(guest_indices)):
            for j in range(i + 1, len(guest_indices)):
                score += interests[guest_indices[i]][guest_indices[j]]

    return score

def repair_individual(individual, table_capacity, no_tables, couples):
    table_counts = {i: 0 for i in range(1, no_tables + 1)}
    for table in individual:
        table_counts[table] += 1

    attempts = 0
    max_attempts = 1000  # A limit to prevent infinite loops

    while any(count > table_capacity for count in table_counts.values()):
        if attempts > max_attempts:
            print("Unable to find a valid configuration within the attempt limit.")
            break
        attempts += 1

        # Find an overfull table
        overfull_table = next(table for table, count in table_counts.items() if count > table_capacity)

        # Try to find a single guest to move
        for guest_index, table in enumerate(individual):
            if table == overfull_table:
                if guest_index + 1 not in couples and (couples.get(guest_index + 1) not in individual or individual[couples[guest_index + 1] - 1] != overfull_table):
                    # This guest is single or their partner is at a different table
                    break
        else:
            # No single guest to move, find any guest
            for guest_index, table in enumerate(individual):
                if table == overfull_table:
                    break

        # Find a table with available capacity
        available_table = next(table for table, count in table_counts.items() if count < table_capacity)

        # Move the guest and their partner if they have one
        partner = couples.get(guest_index + 1)
        individual[guest_index] = available_table
        table_counts[overfull_table] -= 1
        table_counts[available_table] += 1

        if partner:
            partner_index = partner - 1
            individual[partner_index] = available_table
            table_counts[overfull_table] -= 1
            table_counts[available_table] += 1

    return individual



# def repair_individual(individual, table_capacity, no_tables):
#     table_counts = {i: 0 for i in range(1, no_tables + 1)}
#     for table in individual:
#         table_counts[table] += 1

#     overfull_tables = {table: count for table, count in table_counts.items() if count > table_capacity}
#     available_tables = {table: table_capacity - count for table, count in table_counts.items() if count < table_capacity}

#     for table, count in overfull_tables.items():
#         while count > table_capacity:
#             for guest in range(len(individual)):
#                 if individual[guest] == table:
#                     partner = next((p for p, g in couples.items() if g - 1 == guest), None)
#                     for available_table, available_count in available_tables.items():
#                         if available_count > 0:
#                             individual[guest] = available_table
#                             if partner is not None:
#                                 individual[partner - 1] = available_table
#                             available_tables[available_table] -= 1
#                             count -= 1
#                             break
#                     break
#     return individual

def crossover(parent1, parent2, couples):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]

    for guest1, guest2 in couples.items():
        if child1[guest1 - 1] != child1[guest2 - 1]:
            child1[guest2 - 1] = child1[guest1 - 1]
        if child2[guest1 - 1] != child2[guest2 - 1]:
            child2[guest2 - 1] = child2[guest1 - 1]

    return child1, child2


def mutate(individual, couples, no_tables):
    guest1, guest2 = random.sample(range(len(individual)), 2)
    for guest in [guest1, guest2]:
        partner = couples.get(guest + 1, None)
        new_table = random.randint(1, no_tables)
        individual[guest] = new_table
        if partner:
            individual[partner - 1] = new_table
    return individual

population_size = 100
max_generations = 100
mutation_rate = 0.1

population = [[random.randint(1, no_tables) for _ in range(no_guests)] for _ in range(population_size)]
print(couples)
def ga(population, no_guests, no_tables, table_capacity, interests, guests, couples, population_size=100, max_generations=50):
    for generation in range(max_generations):
        fitness_scores = [calculate_fitness(ind, interests, guests, table_capacity, no_tables) for ind in population]
        selected = []
        for _ in range(population_size):
            contenders = random.sample(list(zip(population, fitness_scores)), 3)
            selected.append(max(contenders, key=lambda item: item[1])[0])

        offspring = []
        for _ in range(len(population) // 2):
            parent1, parent2 = select_parents(population, fitness_scores)  # You need to define select_parents
            child1, child2 = crossover(parent1, parent2, couples)
            child1 = mutate(child1, couples, no_tables)
            child2 = mutate(child2, couples, no_tables)
            child1 = repair_individual(child1, table_capacity, no_tables, couples)
            child2 = repair_individual(child2, table_capacity, no_tables, couples)
            offspring.extend([child1, child2])

        population = offspring
        best_fitness = max(fitness_scores)
        best_individual = population[fitness_scores.index(best_fitness)]
        print("Generation ", generation, best_fitness)

    return best_fitness, best_individual


best_score, best_solution = ga(population, no_guests, no_tables, table_capacity, interests, guests, couples)

def print_table_arrangement(best_individual):
    table_assignments = {i: [] for i in range(1, no_tables + 1)}

    for guest_index, table in enumerate(best_individual):
        table_assignments[table].append(str(guest_index + 1))  

    for table, guest_ids in table_assignments.items():
        print(f"Table {table}: {', '.join(guest_ids)}")

print("Max Fitness Score : ", best_score)
print()
print()
print("----------Table Arrangement---------")
print_table_arrangement(best_solution)
