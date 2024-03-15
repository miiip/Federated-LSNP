import time
import Model
import Generator
import Classifier
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits


start = time.time()
iris = load_iris()
x = iris.data
target = iris.target
labels = iris.feature_names
y = []

for i in range(len(target)):
    if target[i] == 0:
        y.append([1, 0, 0])
    elif target[i] == 1:
        y.append([0, 1, 0])
    else:
        y.append([0, 0, 1])
y = np.array(y)

train_x = np.zeros((75, 4))
train_y = np.zeros((75, 3))
test_x = np.zeros((75, 4))
test_y = np.zeros((75, 3))
length = np.load("length_iris.npy")
j = k = 0
for i in range(len(target)):
    if i < 75:
        train_x[j, :] = x[length[i], :]
        train_y[j, :] = y[length[i], :]
        j = j+1
    else:
        test_x[k, :] = x[length[i], :]
        test_y[k, :] = y[length[i], :]
        k = k+1
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

maxgen = 80

generator = Generator.Generator()
kernel_value = generator.kernel_value(train_x[:, :], kernel_option=3)
kernel_value_train = generator.neuron(kernel_value)
classifier = Classifier.Classifier(kernel_value_train, train_y)
model = Model.Model()

for i in range(maxgen):
    model_w, model_l = classifier.train_classification(kernel_value_train, train_y)
    predict = model.classification(model_w, model_l, kernel_value_train, train_y)
    acc_train = model.accuracy_rate(predict, train_y)
    kernel_value_test = generator.kernel_value(test_x[:, :], kernel_option=3)
    predict = model.classification(model_w, model_l, kernel_value_test, test_y)
    acc_test = model.accuracy_rate(predict, test_y)


print("Mean accuracy for training set:", acc_train)
print("Mean accuracy for testing set:", acc_test)
