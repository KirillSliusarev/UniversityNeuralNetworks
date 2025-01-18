from Framework import *
import time

np.random.seed(0)

x = [
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
]
x = np.array(x)
y = [
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
]
y = np.array(y)
hyperparameter_space = {
    'learning_rate': [0.00001, 0.0001, 0.001, 0.01, 0.1],
    'num_layers': [1, 2, 3, 4, 5],
    'num_neurons': [5, 10, 15, 20, 25, 30],
    'act_func': [Sigmoid(), Tanh(), Softmax()],
    'last_func': [Sigmoid(), Tanh(), Softmax()]
}


somemodel = ModelRunner()
somemodel.grid_search_hyperparameters(x,y,x,y,4,10,hyperparameter_space,100)
somemodel.set_best_hparams(4,10,10000)
somemodel.set_train_data(x, y)
somemodel.train()
somemodel.ShowResult(x)
print(somemodel.evaluate(x,y))
print(somemodel.model)
print(somemodel.best_params, somemodel.best_error)
somemodel.set_model(x.shape[1],y.shape[1], 2, 15, last_func=Softmax())
somemodel.set_hyperparameters(num_epoch=10000)
somemodel.set_train_data(x,y)
time.sleep(30)
somemodel.train()
print(somemodel.evaluate(x,y))
print(somemodel.model)
print(somemodel.best_params, somemodel.best_error)


exit()

