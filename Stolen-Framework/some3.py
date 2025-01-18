from Framework import *

np.random.seed(0)

x_train = np.array([
    [58, 160],  # Female
    [60, 165],  # Female
    [80, 180],  # Male
    [75, 175],  # Male
    [50, 155],  # Female
    [90, 185],  # Male
    [70, 170],  # Male
    [40, 150],  # Female
    [65, 168],  # Female
    [85, 182] # Male
])
y_train = np.array([
    [0],  # Female
    [0],  # Female
    [1],  # Male
    [1],  # Male
    [0],  # Female
    [1],  # Male
    [1],  # Male
    [0],  # Female
    [0],  # Female
    [1]  # Male
])

x_val = np.array([
    [58, 160],  # Female
    [60, 165],  # Female
    [80, 180],  # Male
    [75, 175],  # Male
    [50, 155],  # Female
    [90, 185],  # Male
    [70, 170],  # Male
    [40, 150],  # Female
    [65, 168],  # Female
    [85, 182]  # Male
])
y_val = np.array([
    [0],  # Female
    [0],  # Female
    [1],  # Male
    [1],  # Male
    [0],  # Female
    [1],  # Male
    [1],  # Male
    [0],  # Female
    [0],  # Female
    [1]  # Male
])

Sequential([Linear(2, 30), Tanh(), Linear(30, 30), Tanh(), Linear(30, 30), Tanh(), Linear(30, 30), Tanh(), Linear(30, 30), Tanh(), Linear(30, 1), Sigmoid()])



somemodel = ModelRunner()
somemodel.set_model(x_train.shape[1], y_train.shape[1], 5, 30, Tanh(), Sigmoid())
somemodel.set_hyperparameters(learning_rate=0.001, num_epoch=1000)
somemodel.set_train_data(x_train, y_train)
somemodel.train()
somemodel.ShowResult(x_val)
print(somemodel.evaluate(x_val, y_val))
print(somemodel.model)
print(somemodel.best_params, somemodel.best_error)

exit()

