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
        max_val = np.max(self.data, axis=1, keepdims=True)
        exp = np.exp(self.data - max_val)
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
        self.inp = input_count
        self.out = output_count
        weight = np.random.randn(input_count, output_count) * np.sqrt(2.0 / input_count)
        self.weight = Tensor(weight, autograd=True)
        self.bias = Tensor(np.zeros(output_count), autograd=True)
        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, inp):
        return inp.dot(self.weight) + self.bias.expand(0, len(inp.data))

    def __repr__(self):
        return f"Linear({self.inp}, {self.out})"


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
    def __repr__(self):
        return f"Sequential({self.layers})"


class Sigmoid(Layer):
    def forward(self, inp):
        return inp.sigmoid()
    def __repr__(self):
        return "Sigmoid()"



class Tanh(Layer):
    def forward(self, inp):
        return inp.tanh()
    def __repr__(self):
        return "Tanh()"


class Softmax(Layer):
    def forward(self, inp):
        return inp.softmax()
    def __repr__(self):
        return "Softmax()"


class MSELoss(Layer):
    def forward(self, prediction, true_prediction):
        return ((prediction - true_prediction) * (prediction - true_prediction)).sum(0)


class ModelRunner(object):
    def __init__(self):
        self.default_hparams = {
        'learning_rate': [0.00001, 0.0001, 0.001, 0.01, 0.1],
        'num_layers': [1, 2, 3, 4, 5],
        'num_neurons': [5, 10, 15, 20, 25, 30],
        'act_func': [Sigmoid(), Tanh(), Softmax()],
        'last_func': [Sigmoid(), Tanh(), Softmax()]
        }
        self.best_params = None

    def set_model(self, input_size, output_size, num_layers, num_neurons, act_func=Sigmoid(), last_func=Sigmoid()):
        self.layers = []
        for i in range(num_layers):
            self.layers.append(Linear(input_size if i == 0 else num_neurons, num_neurons))
            self.layers.append(act_func)
        self.layers.append(Linear(num_neurons, output_size))
        self.layers.append(last_func)
        self.model = Sequential(self.layers)

    def set_hyperparameters(self, learning_rate=0.01, num_epoch=100, loss=MSELoss()):
        self.sgd = SGD(self.model.get_parameters(), learning_rate)
        self.num_epoch = num_epoch
        self.loss = loss

    def set_train_data(self, x, y, autogradtf=True):
        self.rawx = x
        self.rawy = y
        self.train_x = Tensor(x, autograd=autogradtf)
        self.train_y = Tensor(y, autograd=autogradtf)

    def train(self, epoch_to_show=10):
        if epoch_to_show > self.num_epoch:
            epoch_to_show = self.num_epoch
        for i in range(self.num_epoch):
            self.predictions = self.model.forward(self.train_x)
            self.error = self.loss.forward(self.predictions, self.train_y)
            self.error.backward(Tensor(np.ones_like(self.error.data)))
            self.sgd.step()
            if i % (self.num_epoch / epoch_to_show) == 0 and epoch_to_show != 1:
                print(f"Epoch: {i}/{self.num_epoch}, Error: {self.error}")

    def grid_search_hyperparameters(self, x_train, y_train, x_val, y_val, input_size, output_size, param_grid, num_epochs=100):
        _best_error = float('inf')
        _best_params = None

        for num_layers in param_grid['num_layers']:
            for num_neurons in param_grid['num_neurons']:
                for learning_rate in param_grid['learning_rate']:
                    for act_func in param_grid['act_func']:
                        for last_func in param_grid['last_func']:
                            self.set_model(input_size, output_size, num_layers, num_neurons, act_func, last_func)
                            self.set_hyperparameters(learning_rate=learning_rate, num_epoch=num_epochs)
                            self.set_train_data(x_train, y_train)
                            print(self.model)
                            self.train()
                            _error = self.evaluate(x_val, y_val)[0]


                            if _error < _best_error:
                                _best_error = _error
                                _best_params = (num_layers, num_neurons, learning_rate, act_func, last_func)

        self.best_params = _best_params
        self.best_error = _best_error

    def set_best_hparams(self,input_size, output_size, num_epoch=1000, loss=MSELoss()):
        if self.best_params:
            self.set_model(input_size,output_size,self.best_params[0],self.best_params[1], self.best_params[3], self.best_params[4])
            self.set_hyperparameters(learning_rate=self.best_params[2], num_epoch=num_epoch, loss=loss)
        else: print("There is no best hyperparameters")

    def predictall(self, x_val):
        self.output_layer = self.model.forward(x_val)
        return self.output_layer.data

    def evaluate(self, x_val, y_val):
        self.val_prediction = self.model.forward(Tensor(x_val))
        self.val_loss = self.loss.forward(self.predictions, Tensor(y_val))
        return self.val_loss.data

    def predict(self, inp):
        self.output_layer = self.model.forward(inp)
        return self.output_layer.data

    def ShowResult(self, test_data):
        for inp in test_data:
            print("------------------------------------")
            print(f"предикшн для {inp}:", self.predict(Tensor([inp])))