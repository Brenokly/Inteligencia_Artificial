import numpy as np

# def activation_func(x):
#     return 1 / (1 + np.exp(-x))

# def derivation_func(x):
#     return x * (1 - x)
  
def activation_func(x):
    return np.tanh(x)

def derivation_func(x):
    return 1.0 - np.tanh(x)**2

#EXECUTE
# 1. Inicialização dos pesos sinápticos com valores aleatórios.

input_layer_neurons = 2 # quantidade de neurônios na cama de entrada
hidden_layer_neurons = 8 # quantidade de neurônios na camada oculta
output_neurons = 1 #quantidade de neurônios na camada de saída

hidden_weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons)) #definição de peso aleatório para a camada oculta
output_weights = np.random.uniform(size=(hidden_layer_neurons, output_neurons)) #definição de peso aleatório para a camada de saida

#EXECUTE, ESSA CÉLULA PRECISA SER EXECUTADA
#implementação dos inputs e outputs.

# 2. Aplica o vetor de entradas X1, X2, ... Xn.
inputs = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
outputs = np.array([[1], [0], [0], [1]])

taxa_aprendizagem = 0.2

# EXCECUTE, ESSA CÉLULA É QUEM FAZ O TREINAMENTO
for _ in range(500):  # Número de épocas (iterações de treinamento)
    # 3. Calculam-se os nets dos neurônios da camada oculta, para cada j ε(1,l)
    hidden_layer_activation = np.dot(inputs, hidden_weights)  # Multiplica as entradas pelos pesos da camada oculta

    # 4. Aplica a função de transferência para obter as saídas ij da camada oculta.
    hidden_layer_output = activation_func(hidden_layer_activation)  # Aplica a função sigmóide às ativações da camada oculta

    # 5. Calcula os nets dos neurônios da camada de saída, para cada k ε(1,M)
    output_layer_activation = np.dot(hidden_layer_output, output_weights)  # Multiplica a saída da camada oculta pelos pesos da camada de saída

    # 6. Calcula as saídas Ok dos neurônios da camada de saída.
    predicted_output = activation_func(output_layer_activation)  # Aplica a função sigmóide às ativações da camada de saída

    # 7. Calcula os erros para os neurônios da camada de saída
    error = outputs - predicted_output  # Calcula o erro entre a saída desejada e a saída prevista
    d_predicted_output = error * derivation_func(predicted_output)  # Calcula o gradiente do erro para a saída prevista

    # 8. Calcula-se os erros nos neurônios da camada oculta, para cada j ε(1,l)
    error_hidden_layer = d_predicted_output.dot(output_weights.T)  # Propaga o erro da camada de saída para a camada oculta
    d_hidden_layer = error_hidden_layer * derivation_func(hidden_layer_output)  # Calcula o gradiente do erro para a camada oculta

    # 9. Atualiza-se os pesos da camada de saída.
    output_weights += taxa_aprendizagem * hidden_layer_output.T.dot(d_predicted_output)  # Ajusta os pesos da camada de saída

    # 10. Atualiza-se os pesos da camada oculta.
    hidden_weights += taxa_aprendizagem * inputs.T.dot(d_hidden_layer)  # Ajusta os pesos da camada oculta
        
# Impressão formatada
print("Previsões da rede neural após o treinamento:\n")
for i, (input_val, predicted) in enumerate(zip(inputs, predicted_output)):
    print(f"Entrada {i+1}: {input_val} -> Saída prevista: {predicted[0]:.4f}")