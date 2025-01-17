from Framework import *

# Создаем данные
x = Tensor([
    [9,5,3],#17
    [7,6,2],#15
    [1,2,3],#6
    [9,3,1],#13
    [2,2,2],#6
    [4,5,6]#15
], autograd=True)
y = Tensor([
    17,
    15,
    6,
    13,
    6,
    15
], autograd=True)

# Создаем модель
model = Sequential([
    Linear(3, 10),
    Sigmoid(),
    Linear(10, 10),
    Sigmoid()
])

# Создаем оптимизатор
optimizer = SGD(model.get_parameters(), learning_rate=0.01)

# Обучение модели
for epoch in range(100):
    prediction = model.forward(x)
    loss = MSELoss().forward(prediction, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.data}")