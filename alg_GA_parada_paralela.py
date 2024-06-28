import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
import community
import csv
import time
from multiprocessing import Pool

# Função para ler o arquivo .graph e extrair os dados
def read_graph_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Primeira linha contém o número de vértices e arestas
    vertices = int(lines[0].split()[0])
    edges = []
    
    # As linhas seguintes contêm as conexões entre vértices
    for i in range(1, len(lines)):
        parts = list(map(int, lines[i].split()))
        source = i  # O índice da linha é o nó fonte (ajustado para ser 1-indexado)
        for target in parts:
            edges.append((source, target))

    return vertices, edges

# Função para converter lista de arestas em uma matriz de adjacência
def edges_to_adjacency_matrix(vertices, edges):
    adj_matrix = np.zeros((vertices, vertices))
    for edge in edges:
        i, j = edge
        adj_matrix[i-1][j-1] = 1
        adj_matrix[j-1][i-1] = 1
    return adj_matrix

# Função para calcular a modularidade usando python-louvain
def calculate_modularity_louvain(G):
    partition = community.best_partition(G)
    modularity = community.modularity(partition, G)
    return modularity

# Função de avaliação (fitness function)
def evaluate(individual, adj_matrix):
    # Criar um grafo com base no indivíduo (partição)
    G = nx.from_numpy_array(adj_matrix)
    
    # Inicializar o dicionário community_dict
    community_dict = {}
    
    # Atribuir clusters aos nós
    for node, cluster_id in enumerate(individual):
        community_dict[node] = int(cluster_id)  # Garantir que cluster_id é um inteiro
    
    # Calcular a modularidade
    modularity = community.modularity(community_dict, G)
    
    return modularity,

# Função para executar o algoritmo genético até que 200 gerações passem sem melhora
def run_ea_with_stopping_criteria(pop, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    best = None
    generations_without_improvement = 0
    best_modularity_so_far = float('-inf')  # Inicializa com um valor muito baixo
    
    for gen in range(ngen):
        if generations_without_improvement >= 200:
            break

        # Seleção
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Cruzamento e Mutação
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Avaliar indivíduos com fitness inválido
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Substituir a população atual pelos filhos
        pop[:] = offspring
        
        # Atualizar Hall of Fame
        if halloffame is not None:
            halloffame.update(pop)
        
        # Registrar estatísticas
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        
        current_best = max(logbook.select("max"))
        if current_best > best_modularity_so_far:
            best_modularity_so_far = current_best
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1
        
        if verbose:
            print(logbook.stream)

    return pop, logbook

# Função para executar uma instância do algoritmo genético
def run_ga_instance(instance_num, vertices, edges, adj_matrix):
    # Configuração do algoritmo genético
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(vertices), vertices)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.08)  # Alterar a taxa de mutação aqui
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, adj_matrix=adj_matrix)

    # Estatísticas para registrar durante a execução
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Definir valores específicos para cxpb e mutpb
    cxpb_value = 0.8
    mutpb_value = 0.5
    population = 50
    #inpb = 0.08
    # Execução do algoritmo genético
    pop = toolbox.population(n=population)
    pop, logbook = run_ea_with_stopping_criteria(pop, toolbox, cxpb=cxpb_value, mutpb=mutpb_value, ngen=10000, stats=stats, verbose=False)

    # Rastreando a melhor modularidade encontrada e em qual geração ocorreu
    best_modularity_so_far = float('-inf')  # Inicializa com um valor muito baixo
    best_generation = -1

    for gen, stat in enumerate(logbook):
        max_modularity_gen = stat['max']
        if max_modularity_gen > best_modularity_so_far:
            best_modularity_so_far = max_modularity_gen
            best_generation = gen

    return [population, cxpb_value, mutpb_value, best_modularity_so_far, best_generation] 
start_time = time.time()  # COMECA O TIMER
if __name__ == '__main__':
    # Caminho para o arquivo .graph
    file_path = 'graphs/lesmis.graph'

    # Ler o arquivo e obter o número de vértices e arestas
    vertices, edges = read_graph_file(file_path)

    # Verificação adicional para garantir que os dados foram lidos corretamente
    if vertices == 0 or not edges:
        print("Erro: O arquivo .graph não foi lido corretamente ou está vazio.")
    else:
        # Converter as arestas em uma matriz de adjacência
        adj_matrix = edges_to_adjacency_matrix(vertices, edges)

        

        # Número de execuções paralelas
        num_runs = 10

        # Criar um pool de processos
        with Pool() as pool:
            results = pool.starmap(run_ga_instance, [(i, vertices, edges, adj_matrix) for i in range(num_runs)])

        

        # Abrir o arquivo CSV para escrita
        with open('crit_paralelo_50_008_lesmis_1.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            # Escrever o cabeçalho
            writer.writerow(["POP", "CXPB", "MUTPB", "BEST_MODULARITY", "BEST_GEN"])
            # Escrever os resultados de cada execução
            for result in results:
                writer.writerow(result)
        end_time = time.time()  # ACABA TIMER
        execution_time = end_time - start_time
        print(f"\nTempo de execução: {execution_time} segundos")
