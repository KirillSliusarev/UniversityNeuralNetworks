import numpy as np


class Tensor(object):
    grad = None

    # children = {}
    def __init__(self, data, creators=None, operation_on_creation=None, autograd=False, id=None):
        self.data = np.array(data)
        self.creators = creators
        self.operation_on_creation = operation_on_creation
        self.autograd = autograd
        # self.grad = None
        self.children = {}

        if id is None:
            self.id = np.random.randint(0, 1000000000)

        if (self.creators is not None):
            for creator in creators:
                if self.id not in creator.children:
                    creator.children[self.id] = 1
                else:
                    creator.children[self.id] += 1

    def __add__(self, other):
        if self.autograd:
            return Tensor(self.data + other.data, [self, other], "+", True)
        else:
            return Tensor(self.data + other.data)

    def __str__(self):
        # return str(self.data.__str__())
        return str(self.data)

    def backward(self, grad=None, grad_child=None):
        if self.autograd:
            if grad is None:
                grad = Tensor(np.ones_like(self.data))
            if grad_child is not None:
                if (self.children[grad_child.id]) > 0:
                    self.children[grad_child.id] -= 1
            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

            if (self.creators is not None and (self.check_grads_from_child() or grad_child is None)):
                match self.operation_on_creation:
                    case "+":
                        self.creators[0].backward(self.grad, self)
                        self.creators[1].backward(self.grad, self)

                    case "-1":
                        self.creators[0].backward(self.grad.__neg__(), self)

                    case "-":
                        self.creators[0].backward(self.grad, self)
                        self.creators[1].backward(self.grad.__neg__(), self)

                    case str(op) if "sum" in op:
                        axis = int(op.split("_")[1])
                        self.creators[0].backward(self.grad.expand(axis, self.creators[0].data.shape[axis]), self)

                    case str(op) if "expand" in op:
                        axis = int(op.split("_")[1])
                        self.creators[0].backward(self.grad.sum(axis), self)

                    case "*":
                        self.creators[0].backward(self.grad * self.creators[1], self)
                        self.creators[1].backward(self.grad * self.creators[0], self)

                    case str(op) if "dot" in op:
                        temp = self.grad.dot(self.creators[1].transpose())
                        self.creators[0].backward(temp, self)
                        temp = self.grad.transpose().dot(self.creators[0]).transpose()
                        self.creators[1].backward(temp, self)

                    case str(op) if "transpose" in op:
                        self.creators[0].backward(self.grad.transpose(), self)

                    case str(op) if "sigmoid" in op:
                        temp = Tensor(np.ones_like(self.grad.data))
                        self.creators[0].backward(self.grad * (self * (temp - self)), self)

                    case str(op) if "tanh" in op:
                        temp = Tensor(np.ones_like(self.grad.data))
                        self.creators[0].backward(self.grad * (temp - (self * self)), self)

                    case str(op) if "softmax" in op:
                        self.creators[0].backward(Tensor(self.grad.data), self)

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1, [self], "-1", True)
        else:
            return Tensor(self.data * -1)

    def __sub__(self, other):
        if self.autograd:
            return Tensor(self.data - other.data, [self, other], "-", True)
        else:
            return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd:
            return Tensor(self.data * other.data, [self, other], "*", True)
        else:
            return Tensor(self.data * other.data)

    def sum(self, axis):
        if self.autograd:
            return Tensor(self.data.sum(axis), [self], "sum_" + str(axis), True)
        return Tensor(self.data.sum(axis))

    def expand(self, axis, count_copies):
        transpose = list(range(0, len(self.data.shape)))
        transpose.insert(axis, len(self.data.shape))
        # print(transpose)
        expand_shape = list(self.data.shape) + [count_copies]
        expand_data = self.data.repeat(count_copies).reshape(expand_shape)
        expand_data = expand_data.transpose(transpose)
        if (self.autograd):
            return Tensor(expand_data, [self], "expand_" + str(axis), autograd=True)
        return Tensor(expand_data)

    def dot(self, other):
        if self.autograd:
            return Tensor(self.data.dot(other.data), [self, other], "dot", True)
        else:
            return Tensor(self.data.dot(other.data))

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(), [self], "transpose", True)
        else:
            return Tensor(self.data.transpose())

    def sigmoid(self):
        if self.autograd:
            return Tensor(1 / (1 + np.exp(-self.data)), [self], "sigmoid", True)
        else:
            return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data), [self], "tanh", True)
        else:
            return Tensor(np.tanh(self.data))

    def softmax(self):
        exp = np.exp(self.data)
        exp = exp / np.sum(exp, axis=1, keepdims=True)
        if self.autograd:
            return Tensor(exp, [self], "softmax", True)
        return Tensor(exp)

    def check_grads_from_child(self):
        for id in self.children:
            if self.children[id] != 0:
                return False
        return True

    def __repr__(self):
        return str(self.data.__repr__())


class SGD(object):
    def __init__(self, weigts, learning_rate):
        self.weights = weigts
        self.learning_rate = learning_rate

    def step(self):
        for weight in self.weights:
            weight.data -= self.learning_rate * weight.grad.data
            weight.grad.data *= 0


class Layer(object):
    def __init__(self):
        self.parameters = []

    def get_parameters(self):
        return self.parameters


class Linear(Layer):
    def __init__(self, input_count, output_count):
        super().__init__()
        weight = np.random.randn(input_count, output_count) * np.sqrt(2.0 / input_count)
        self.weight = Tensor(weight, autograd=True)
        self.bias = Tensor(np.zeros(output_count), autograd=True)
        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, inp):
        return inp.dot(self.weight) + self.bias.expand(0, len(inp.data))


class Sequential(Layer):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inp):
        for layer in self.layers:
            inp = layer.forward(inp)
        return inp

    def get_parameters(self):
        params = []
        for layer in self.layers:
            params += layer.get_parameters()
        return params


class Sigmoid(Layer):
    def forward(self, inp):
        return inp.sigmoid()


class Tanh(Layer):
    def forward(self, inp):
        return inp.tanh()


class Softmax(Layer):
    def forward(self, inp):
        return inp.softmax()


class MSELoss(Layer):
    def forward(self, prediction, true_prediction):
        return ((prediction - true_prediction) * (prediction - true_prediction)).sum(0)


np.random.seed(0)

x = Tensor([
    [0, 0, 0, 0],  # 0
    [0, 0, 0, 1],  # 1
    [0, 0, 1, 0],  # 2
    [0, 0, 1, 1],  # 3
    [0, 1, 0, 0],  # 4
    [0, 1, 0, 1],  # 5
    [0, 1, 1, 0],  # 6
    [0, 1, 1, 1],  # 7
    [1, 0, 0, 0],  # 8
    [1, 0, 0, 1]  # 9
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

model = Sequential([Linear(4, 15), Sigmoid(), Linear(15, 10), Softmax()])
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
    [0, 0, 0, 0],  # 0
    [0, 0, 0, 1],  # 1
    [0, 0, 1, 0],  # 2
    [0, 0, 1, 1],  # 3
    [0, 1, 0, 0],  # 4
    [0, 1, 0, 1],  # 5
    [0, 1, 1, 0],  # 6
    [0, 1, 1, 1],  # 7
    [1, 0, 0, 0],  # 8
    [1, 0, 0, 1]  # 9
])

for inp in x:
    print("------------------------------------")
    print(f"Предсказанная цифра для {inp}:", predict(Tensor([inp])))
