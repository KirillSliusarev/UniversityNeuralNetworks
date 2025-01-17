from Framework import *

# Определяем входные данные и целевые значения для XOR
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Создаем экземпляр ModelRunner
model_runner = ModelRunner()

# Устанавливаем модель с одним скрытым слоем и двумя нейронами
model_runner.set_model(input_size=2, output_size=1, num_layers=1, num_neurons=2)

# Устанавливаем гиперпараметры
model_runner.set_hyperparameters(learning_rate=0.1, num_epoch=10000)

# Устанавливаем тренировочные данные
model_runner.set_train_data(x_train, y_train)

# Обучаем модель
model_runner.train(epoch_to_show=1000)

# Проверяем результаты на тренировочных данных
print("Training results:")
model_runner.ShowResult(x_train)

# Проверяем результаты на тестовых данных (в данном случае они совпадают с тренировочными)
print("Test results:")
model_runner.ShowResult(x_train)