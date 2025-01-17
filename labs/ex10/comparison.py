import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return (x > 0) * x

def relu_deriv(x):
    return x > 0

# Входные данные и ожидаемый прогноз
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Параметры нейронной сети
input_size = len(x[0])
hidden_size = 4
output_size = len(y[0])

# Фиксируем генератор случайных чисел
np.random.seed(1)
weight_hid = np.random.uniform(size=(input_size, hidden_size))
weight_out = np.random.uniform(size=(hidden_size, output_size))

learning_rate = 0.1
epochs = 10000

# Функция для обучения сети
def train_network(activation, activation_deriv, hidden_size):
    weight_hid = np.random.uniform(size=(input_size, hidden_size))
    weight_out = np.random.uniform(size=(hidden_size, output_size))

    for epoch in range(epochs):
        layer_hid = activation(np.dot(x, weight_hid))
        layer_out = activation(np.dot(layer_hid, weight_out))
        error = (layer_out - y) ** 2

        layer_out_delta = (layer_out - y) * activation_deriv(layer_out)
        layer_hidden_delta = layer_out_delta.dot(weight_out.T) * activation_deriv(layer_hid)

        weight_out -= learning_rate * layer_hid.T.dot(layer_out_delta)
        weight_hid -= learning_rate * x.T.dot(layer_hidden_delta)


    new_input = np.array([[0,1]])
    layer_hid = activation(new_input.dot(weight_hid))
    layer_out = activation(layer_hid.dot(weight_out))

    print("Предсказание: ", layer_out)

# Сравнение функций активации
print("Sigmoid:")
train_network(sigmoid, sigmoid_deriv, hidden_size)

print("Tanh:")
train_network(tanh, tanh_deriv, hidden_size)

print("ReLU:")
train_network(relu, relu_deriv, hidden_size)

# Сравнение количества нейронов в скрытом слое
hidden_sizes = [2, 4, 8, 16]
for hidden_size in hidden_sizes:
    print(f"Hidden size: {hidden_size}")
    train_network(relu, relu_deriv, hidden_size)


# Функции активации:
#
# Sigmoid: Может иметь проблемы с градиентным исчезанием, особенно при глубоких сетях.
# Tanh: Обычно работает лучше, чем sigmoid, так как выходные значения центрированы вокруг нуля.
# ReLU: Обычно обеспечивает быструю сходимость и решает проблему градиентного исчезания, но может страдать от "умирающих ReLU" (dead ReLU), когда нейроны перестают активироваться.
# Количество нейронов в скрытом слое:
#
# Увеличение количества нейронов в скрытом слое может улучшить точность, но также может увеличить время обучения и риск переобучения.
# Оптимальное количество нейронов зависит от сложности задачи и количества данных.
# Вывод
# Для задачи XOR, ReLU обычно обеспечивает лучшую сходимость и точность. Увеличение количества нейронов в скрытом слое может улучшить точность, но следует быть осторожным с переобучением.
