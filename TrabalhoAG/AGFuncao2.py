import numpy as np
import matplotlib.pyplot as plt

# Função de aptidão
def fitnessfunction(x, y):
    return np.sqrt(x**3 + 2*y**4)

def crossover(parent1, parent2):
    num_bits = 3 # Cada cromossomo é representado por 3 bits
    child1 = np.zeros_like(parent1)
    child2 = np.zeros_like(parent2)

    for i in range(len(parent1)):
        # Transformando os inteiros em suas representações binárias (3 bits)
        p1_bits = np.array([int(b) for b in format(parent1[i], '03b')])
        p2_bits = np.array([int(b) for b in format(parent2[i], '03b')])
        
        # Escolhe um ponto de crossover aleatório entre 1 e 2
        crossover_point = np.random.randint(1, num_bits)
        
        # Realiza o crossover
        c1_bits = np.concatenate((p1_bits[:crossover_point], p2_bits[crossover_point:]))
        c2_bits = np.concatenate((p2_bits[:crossover_point], p1_bits[crossover_point:]))
        
        # Converte de volta os bits para inteiros
        child1[i] = int(''.join(map(str, c1_bits)), 2)
        child2[i] = int(''.join(map(str, c2_bits)), 2)
        
        return child1, child2

def mutate(child, mutation_probability):
    num_bits = 3  # Cada cromossomo é representado por 3 bits
    
    for i in range(len(child)):
        # Converter o gene para a sua representação binária de 3 bits
        bits = list(format(child[i], f'0{num_bits}b'))
        
        for j in range(num_bits):
            if np.random.rand() < mutation_probability:
                # Escolher um valor aleatório para o bit (0 ou 1)
                bits[j] = str(np.random.randint(0, 2))
        
        # Converter a representação binária de volta para inteiro
        child[i] = int(''.join(bits), 2)
        
        return child

# Função para girar a roleta e selecionar um índice, evitando um índice excluído
def spin(cumulative_intervals, excluded_index=None):
    while True:
        spin_value = np.random.uniform(0, 360) # Gerar um valor aleatório entre 0 e 360
        for i, limit in enumerate(cumulative_intervals): # Para cada índice e limite
            if spin_value <= limit: # Se o valor gerado for menor ou igual ao limite
                if excluded_index is not None and i == excluded_index: # Se o índice for o índice excluído
                    break # Pular para a próxima iteração
                return i # Retornar o índice

# Inicialização da população
population = np.array([[5, 2], [6, 4], [1, 5], [7, 3]])
mutation_probability = 0.05 # Probabilidade de mutação

# Aplicando a função de aptidão para toda a população
fitness_values = fitnessfunction(population[:, 0], population[:, 1]) # Calcular a aptidão de cada indivíduo
fitness_values = np.where(fitness_values == 0, 1e-10, fitness_values) # Substituir valores 0 por 1e-10
total_fitness = np.sum(1 / fitness_values) # Calcular a aptidão total
selection_probabilities = (1 / fitness_values) / total_fitness # Calcular a probabilidade de seleção de cada indivíduo

all_fitness_values = [] # Lista para armazenar os valores de aptidão de todos os indivíduos

# Inicializando o contador de iterações
iterations = 0

while not np.any(selection_probabilities > 0.99): # Enquanto nenhum indivíduo tiver probabilidade de seleção maior que 99%
    
    # Criando um dicionário que mapeia população -> (fitness_value, probabilidade de seleção)
    fitness_map = {
      tuple(individual): (fitness, prob)
      for individual, fitness, prob in zip(population, fitness_values, selection_probabilities)
    }

    # Exibindo o dicionário
    for individual, (fitness, prob) in fitness_map.items():
      individual_str = f"({int(individual[0])}, {int(individual[1])})"
      print(f"Indivíduo (x,y): {individual_str}, Aptidão: {fitness:.2f}, Probabilidade de Seleção: {(prob * 100):.2f} %")
    
    all_fitness_values.append(np.min(fitness_values)) # Armazenar os valores de aptidão de todos os indivíduos
    
    intervals = selection_probabilities * 360 # Calcular os intervalos de seleção
    cumulative_intervals = np.cumsum(intervals) # Calcular os intervalos acumulados
    
    parent1_index = spin(cumulative_intervals) # Selecionar o índice do pai 1
    parent1 = population[parent1_index] # Selecionar o pai 1
    
    parent2_index = spin(cumulative_intervals, excluded_index=parent1_index) # Selecionar o índice do pai 2 (excluindo o pai 1)
    parent2 = population[parent2_index] # Selecionar o pai 2
    
    # Exibindo os pais selecionados
    print(f"\nPai 1: ({int(parent1[0])}, {int(parent1[1])})")
    print(f"Pai 2: ({int(parent2[0])}, {int(parent2[1])})")
    
    child1, child2 = crossover(parent1, parent2) # Aplicar o crossover gerando dois filhos
    
    # Exibindo os filhos gerados
    print(f"\nFilho 1: ({int(child1[0])}, {int(child1[1])})")
    print(f"Filho 2: ({int(child2[0])}, {int(child2[1])})")
    
    child1 = mutate(child1, mutation_probability) # Aplicar a mutação no filho 1
    child2 = mutate(child2, mutation_probability) # Aplicar a mutação no filho 2
    
    # Exibindo os filhos gerados pela mutação
    print(f"\nFilho pós mutação 1: {child1}")
    print(f"Filho pós mutação 2: {child2}\n")
    
    print("--------------------------------------------------")
    
    # Atualizar a população: remover os pais e adicionar os filhos
    population = np.array([indiv for i, indiv in enumerate(population) if i not in [parent1_index, parent2_index]])
    population = np.vstack([population, [child1, child2]]) # Adicionar os filhos gerados
    
    fitness_values = fitnessfunction(population[:, 0], population[:, 1]) # Calcular a aptidão de cada indivíduo
    fitness_values = np.where(fitness_values == 0, 1e-10, fitness_values) # Substituir valores 0 por 1e-10
    total_fitness = np.sum(1 / fitness_values) # Calcular a aptidão total
    selection_probabilities = (1 / fitness_values) / total_fitness # Calcular a probabilidade de seleção de cada indivíduo
    
    iterations += 1 # Incrementar o contador de iterações
    
all_fitness_values.append(np.min(fitness_values)) # Armazenar os valores do último indivíduo

# Encontrar o indivíduo com a maior probabilidade de seleção
max_prob_index = np.argmax(selection_probabilities) # Índice do indivíduo com maior probabilidade de seleção
best_individual = population[max_prob_index] # Melhor indivíduo

print(f"\nNúmero de iterações: {iterations}")
print(f"Indivíduo que alcançou probabilidade de 100%: ({int(best_individual[0])}, {int(best_individual[1])})")

# Definindo o número de gerações a cada 5 iterações
generations = list(range(0, iterations + 1, 5))

# Selecionando os valores de aptidão a cada 5 iterações
fitness_at_generations = all_fitness_values[::5]

# Verificando se o último valor foi incluído
if len(all_fitness_values) - 1 not in range(0, iterations + 1, 5):
    generations.append(iterations)  # Adiciona a última geração
    fitness_at_generations.append(all_fitness_values[-1])  # Adiciona o último valor de aptidão

# Plotando o gráfico
plt.figure(figsize=(10, 6))
plt.plot(generations, fitness_at_generations, marker='o', linestyle='-', color='b', label='Fitness a cada 5 gerações')
plt.axhline(y=0, color='g', linestyle='--', label='Valor ideal (0)')

plt.title('Convergência dos Valores de Fitness a cada 5 Gerações')
plt.xlabel('Geração')
plt.ylabel('Valor de Fitness')
plt.legend()
plt.grid(True)
plt.show()