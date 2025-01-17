from Framework import *

np.random.seed(0)

x = Tensor([
    [0,0,0,0], # 0
    [0,0,0,1], # 1
    [0,0,1,0], # 2
    [0,0,1,1], # 3
    [0,1,0,0], # 4
    [0,1,0,1], # 5
    [0,1,1,0], # 6
    [0,1,1,1], # 7
    [1,0,0,0], # 8
    [1,0,0,1]  # 9
], autograd=True)

y = Tensor([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ], autograd=True)


model = Sequential([Linear(4,15), Sigmoid(), Linear(15,10), Softmax()])
sgd = SGD(model.get_parameters(), 0.01)
num_epoch = 10000

loss = MSELoss()
for i in range(num_epoch):
    predictions = model.forward(x)
    error = loss.forward(predictions, y)
    error.backward(Tensor(np.ones_like(error.data)))
    sgd.step()
    if num_epoch % 1000 == 0:
        print(f"Epoch: {num_epoch}, Error: {error}")

def predict(inp):
    output_layer = model.forward(inp)
    return np.argmax(output_layer.data)
x = ([
    [0,0,0,0], # 0
    [0,0,0,1], # 1
    [0,0,1,0], # 2
    [0,0,1,1], # 3
    [0,1,0,0], # 4
    [0,1,0,1], # 5
    [0,1,1,0], # 6
    [0,1,1,1], # 7
    [1,0,0,0], # 8
    [1,0,0,1]  # 9
])

for inp in x:
    print("------------------------------------")
    print(f"Предсказанная цифра для {inp}:", predict(Tensor([inp])))