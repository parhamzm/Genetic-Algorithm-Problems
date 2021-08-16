import numpy as np
import math
import copy
import random
from itertools import islice
import matplotlib.pyplot as plt


# Extract m from first line of the input file:
def extract_m_from_input_file(file_name='input30.txt'):
    with open(file_name, 'r') as f:
        first_line = f.readline()
        m = int(first_line)
    return m


# Extract the input matrix from the input file:
def extract_input_matrix_from_input_file(file_name='input30.txt'):
    with open(file_name, 'r') as fin:
        mat = []
        for line in islice(fin, 1, None): # to start reading the file from line 2 to the end
            line = line.strip()
            l = [int(num.rstrip().strip()) for num in line.split(' ')]
            mat.append(l)
    return mat

  
def initialize_population(population_size=100, m=0, mat=[]):
    population = []
    for i in range(0, population_size):
        try:
            tmp = random.sample(range(0, len(mat[0])-1), m)
        except ValueError:
            print('Sample size exceeded population size.')
        population.append(tmp)
    return population


def mutate(instance, mutate_items, input_mat=[]):
    size = len(mutate_items)
    n = len(instance)
    genes = list(range(0, len(input_mat[0])))
    remained_genes = [x for x in genes if x not in instance]
    randomlist = random.sample(remained_genes, size)
    new_instance = copy.deepcopy(instance)
    i = 0
    for index, item in enumerate(new_instance):
        if item in mutate_items:
            if i == len(randomlist):
                break
            new_instance[index] = randomlist.__getitem__(i)
            i+=1
    mutated = copy.deepcopy(new_instance)
    return mutated


def mutate_population_randomly(population, mutation_rate=0.5):
    # Apply random mutation
    population_tmp = copy.deepcopy(population)
    population_tmp = np.array(population_tmp)
    random_mutation_array = np.random.random(
        size=(population_tmp.shape))
    random_mutation_boolean = random_mutation_array <= mutation_rate

    for i in range(0, len(population_tmp)):
        if population_tmp[i][random_mutation_boolean[i]].size != 0:
            temp_instance = population_tmp[i][random_mutation_boolean[i]].tolist()
            mutated = mutate(instance=population_tmp[i].tolist(), 
                                mutate_items=temp_instance, 
                                input_mat=input_mat)
            population[i] = mutated
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


def fitness_function(instance, mat=[], m=0):
    instance_copy = copy.deepcopy(instance)
    total = len(instance)
    score = 0
    for item in instance_copy:
        breaked = 0
        new_instance = copy.deepcopy(instance_copy)
        if type(new_instance) is not list:
            new_instance = new_instance.tolist()
        new_instance.remove(item)
        for i in new_instance:
            if mat[item][i] == 1:
                breaked = 1
                break
        if breaked == 0:
            score+=1
    fitness = score / total

    return fitness

    # for row in range(1, len(mat[0])-1):
    #     for col in range(row+1, len(mat[0])):
    #         pass
    
def calculate_population_fitness(population, input_mat=[], m=0):
    population_fitness_scores = []
    for person in population:
        score = fitness_function(instance=person, mat=input_mat, m=m)
        population_fitness_scores.append(score)
    return population_fitness_scores


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


def save_result_to_file(file_name='output.txt', input_array=[]):
    # Displaying the array
    print('Array:\n', input_array)
    
    # Saving the 2D array in a text file
    np.savetxt(fname=file_name, X=input_array, fmt="%d")


# The main Part of Genetic Algorithm:
def main(mutation_rate=0.02, POPULATION_SIZE=0, 
        MAXIMUM_GENERATION=0, CHROMOSOME_LENGTH=0,
        m=0, input_mat=[]):
    # Create starting population
    best_person = None
    best_score_progress = [] # Tracks fitness progress
    avg_score_progress = [] # Tracks avg progress
    diversity_tracking = [] # Tracks diversity
    population = initialize_population(population_size=POPULATION_SIZE, 
                                        m=CHROMOSOME_LENGTH, mat=input_mat)
    init_scores = calculate_population_fitness(population, input_mat=input_mat, m=m)
    print("Init Score: ", init_scores)
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
        
        population = mutate_population_randomly(population=population, 
                                            mutation_rate=mutation_rate)

        # Score best solution, and add to tracker
        init_scores = calculate_population_fitness(population=population, 
                                                input_mat=input_mat, m=m)
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
    print ('End best score, percent target:=> %.1f' %best_score)
    print("The Output of our App is:=> ", best_person)
    save_result_to_file(input_array=best_person)
    plot_result(best_score_progress, y_text="Best Score in Generation (%)", plot_title="Maximum Fitness in Each Generation")
    plot_result(avg_score_progress, y_text="Average Score in Generation (%)", plot_title="Average Fitness in Each Generation", max_iter=300)
    plot_diversity_result(diversity_tracking, max_iter=205, max_pop=POPULATION_SIZE)
    

if __name__ == '__main__':
    # Set general parameters
    # Number of individuals in each generation
    POPULATION_SIZE = 100
    MAXIMUM_GENERATION = 300
    m = extract_m_from_input_file(file_name='input60.txt')
    input_mat = extract_input_matrix_from_input_file(
        file_name='input60.txt')
    CHROMOSOME_LENGTH = m
    main(mutation_rate=0.02, POPULATION_SIZE=POPULATION_SIZE, 
        MAXIMUM_GENERATION=MAXIMUM_GENERATION, 
        CHROMOSOME_LENGTH=CHROMOSOME_LENGTH,
        m=m, input_mat=input_mat)