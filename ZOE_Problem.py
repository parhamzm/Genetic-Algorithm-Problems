import numpy as np
import math
import copy
import random
from itertools import islice
import matplotlib.pyplot as plt
import os

# Extract the input matrix from the input file:
def extract_input_matrix_from_input_file(file_name='q2.txt'):
    cwd = os.getcwd()  # Get the current working directory (cwd)
    files = os.listdir(cwd)  # Get all the files in that directory
    # print("Files in %r: %s" % (cwd, files))
    __location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(file_name)))
    # print("Location: ", __location__)
    with open(file_name, 'r') as fin:
        mat = []
        for line in islice(fin, 0, None): # to start reading the file from line 2 to the end
            line = line.strip()
            l = [int(num.rstrip().strip()) for num in line.split(' ')]
            mat.append(l)
    return mat

# Function to create the initialization population:
def initialize_population(population_size=100, chromosome_size=20):
    # Set up an initial array of all zeros
    population = np.zeros((population_size, chromosome_size), dtype=int)
    # Loop through each row (individual)
    for i in range(population_size):
        # Choose a random number of ones to create
        ones = random.randint(0, chromosome_size)
        # Change the required number of zeros to ones
        population[i, 0:ones] = 1
        # Sfuffle row
        np.random.shuffle(population[i])
    return population


def fitness_function(instance, input_mat=[], result_mat_length=0):
    instance_copy = copy.deepcopy(instance)
    total = result_mat_length
    score = 0
    if not isinstance(instance, (np.ndarray)):
        instance = np.array(instance)
    if not isinstance(input_mat, (np.ndarray)):
        input_mat = np.array(input_mat)
    res_mat = input_mat.dot(instance)
    res_one_matrix = np.ones(result_mat_length).reshape(-1)
    number_of_result_ones = np.count_nonzero(res_mat==1)
    score = number_of_result_ones
    fitness = score / total
    return fitness


def calculate_population_fitness(population, input_mat=[], m=0):
    population_fitness_scores = []
    for person in population:
        score = fitness_function(instance=person, input_mat=input_mat, result_mat_length=m)
        population_fitness_scores.append(score)
    return population_fitness_scores


def mutate_population(population=[], mutation_rate=0.01):
    # Apply random mutation
    population = np.array(population)
    random_mutation_array = np.random.random(
        size=(population.shape))
    random_mutation_boolean = random_mutation_array <= mutation_rate
    population[random_mutation_boolean] = np.logical_not(population[random_mutation_boolean])
    
    # Return mutation population
    return population


def crossover(instance1, instance2):
    # Get length of chromosome (m)
    chromosome_length = len(instance1)

    # Pick crossover point, avoding ends of chromsome
    crossover_point = random.randint(1, chromosome_length-1)

    # Create children. np.hstack joins two arrays
    child_1 = np.hstack((instance1[0:crossover_point],
                        instance2[crossover_point:]))
    
    child_2 = np.hstack((instance2[0:crossover_point],
                        instance1[crossover_point:]))

    # Return children
    return child_1, child_2


def select_individual_by_tournament(population, scores):
    # Get population size
    population_size = len(scores)
    
    # Pick individuals for tournament
    fighter_1 = random.randint(0, population_size-1)
    fighter_2 = random.randint(0, population_size-1)
    
    # Get fitness score for each
    fighter_1_fitness = scores[fighter_1]
    fighter_2_fitness = scores[fighter_2]
    
    # Identify undividual with highest fitness
    # Fighter 1 will win if score are equal
    if fighter_1_fitness >= fighter_2_fitness:
        winner = fighter_1
    else:
        winner = fighter_2
    
    # Return the chromsome of the winner
    return population[winner]


def save_result_to_file(file_name='output.txt', input_array=[]):
    # Displaying the array
    # print('Array:\n', input_array)
    
    # Saving the 2D array in a text file
    np.savetxt(fname=file_name, X=input_array, fmt="%d")


def plot_result(items_list, y_text="Best Score in Generation (%)", plot_title="Maximum Fitness in Each Generation", max_iter=105):
    plt.rcParams["figure.figsize"] = (25,8)

    fig, ax = plt.subplots()


    plt.ylim(17,105)
    plt.xlim(0, max_iter)
    plt.plot((items_list), marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Generation', fontsize=13)
    plt.xticks(np.arange(0, max_iter, step=10)) #change from 0-based array index to 1-based human-readable label
    plt.ylabel(y_text, fontsize=13)
    plt.title(plot_title, fontsize=18)
    # plt.axvline(x=best_choice_pca, color="k", linestyle="--")
    plt.axhline(y=100, color='r', linestyle='-')
    plt.text(10, 95, '100% cut-off threshold', color = 'red', fontsize=16)

    ax.grid(axis='x')
    plt.show()


def plot_diversity_result(items_list, y_text="Diversity (#)", 
                plot_title="Diversity in Each Generation", max_iter=105, max_pop=100):
    plt.rcParams["figure.figsize"] = (25,8)

    fig, ax = plt.subplots()


    plt.ylim(0, max_pop)
    plt.xlim(0, max_iter)
    plt.plot((items_list), marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Generation', fontsize=13)
    plt.xticks(np.arange(0, max_iter, step=10)) #change from 0-based array index to 1-based human-readable label
    plt.ylabel(y_text, fontsize=13)
    plt.title(plot_title, fontsize=18)

    ax.grid(axis='x')
    plt.show()


def main(mutation_rate=0.02, POPULATION_SIZE=0, 
        MAXIMUM_GENERATION=0, CHROMOSOME_LENGTH=0,
        m=0, input_mat=[]):
    # Create starting population
    best_person = None
    best_score_progress = [] # Tracks fitness progress
    avg_score_progress = [] # Tracks avg progress
    diversity_tracking = [] # Tracks diversity
    population = initialize_population(population_size=POPULATION_SIZE, 
                                        chromosome_size=CHROMOSOME_LENGTH)
    init_scores = calculate_population_fitness(population, input_mat=input_mat, m=m)
    # print("Init Score: ", init_scores)
    best_score = np.max(init_scores) * 100
    print ('Starting best score, percent target: %.2f' %best_score)
    # Add starting best score to progress tracker
    best_score_progress.append(best_score)

    for idx, generation in enumerate(range(MAXIMUM_GENERATION)):
        # Create an empty list for new population
        new_population = []
        # Create new popualtion generating two children at a time
        for i in range(int(POPULATION_SIZE/2)):
            parent_1 = select_individual_by_tournament(population, init_scores)
            parent_2 = select_individual_by_tournament(population, init_scores)
            child_1, child_2 = crossover(parent_1, parent_2)
            new_population.append(child_1)
            new_population.append(child_2)

        # Replace the old population with the new one
        population = copy.deepcopy(new_population) #np.array(new_population)
        # Apply mutation
        if idx % 50 == 0:
            mutation_rate = mutation_rate / 2
        population = mutate_population(population=population, mutation_rate=mutation_rate)

        # Score best solution, and add to tracker
        init_scores = calculate_population_fitness(population=population, input_mat=input_mat, m=m)
        best_score = np.max(init_scores) * 100
        best_score_column = np.argmax(init_scores, axis=0)
        avg_score = np.average(init_scores, axis=0) * 100
        best_person = population[best_score_column]
        best_score_progress.append(best_score)
        avg_score_progress.append(avg_score)
        (unique, counts) = np.unique(population, return_counts=True, axis=0)
        population_diversity = len(counts)
        diversity_tracking.append(population_diversity)
        print("=======================================================")
        print("| *************>>> Generation : {0:d} <<<************* |".format(idx+1))
        print ('| Generation Best score, percent target:====> {0:.2f}% |'.format(best_score))
        print ('| Generation Average score, percent target:=> {0:.2f}%  |'.format(avg_score))
        print("|        Generation Diversity Count:===> {0:d}         |".format(population_diversity))
        print("=======================================================")
    # GA has completed required generation
    print("=======================================================")
    print ('|| End best score, percent target: %.1f ||' %best_score)
    print("|| The Output of our App is:=> {} ||".format(best_person))
    plot_result(best_score_progress, y_text="Best Score in Generation (%)", 
        plot_title="Maximum Fitness in Each Generation")
    plot_result(avg_score_progress, y_text="Average Score in Generation (%)", 
        plot_title="Average Fitness in Each Generation", max_iter=MAXIMUM_GENERATION)
    plot_diversity_result(diversity_tracking, max_iter=MAXIMUM_GENERATION, max_pop=POPULATION_SIZE)
    save_result_to_file(input_array=best_person)

if __name__ == '__main__':
    # Set Hyperparameters:
    POPULATION_SIZE = 1000
    MAXIMUM_GENERATION = 30
    # Import Matrix:
    input_mat = extract_input_matrix_from_input_file(file_name='q2.txt')
    column_one = [i[0] for i in input_mat]
    n = len(input_mat[0])
    CHROMOSOME_SIZE = n
    m = len(column_one)
    print("N(Number of Columns):=> ", n)
    print("M(Number of Rows):=> ", m)
    main(mutation_rate=0.01, POPULATION_SIZE=POPULATION_SIZE, 
        MAXIMUM_GENERATION=MAXIMUM_GENERATION, CHROMOSOME_LENGTH=CHROMOSOME_SIZE,
        m=m, input_mat=input_mat)

# x = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 1, 0, 0, 0, 0, 0, 0,0, 0, 0])
# mat = extract_input_matrix_from_input_file(file_name='q2 input 41 40-2.txt')
# mat = np.array(mat)
# print(mat.dot(x))