import time
import Model
import Generator
import Classifier
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

def add_synthetic_data(x, y, num_samples, noise_level):
    for idx in range(num_samples):
        sample = x[idx%len(x)]
        noise = np.random.normal(loc=0, scale=1, size=sample.shape) * noise_level
        sample = sample + noise
        x = np.vstack([x, sample])
        y = np.vstack([y, y[idx]])
    return x, y

def generate_synthetic_data(x, noise_level):
    for idx in range(len(x)):
        sample = x[idx%len(x)]
        noise = np.random.normal(loc=0, scale=1, size=sample.shape) * noise_level
        sample = sample + noise
        x[idx] = sample
    return x

def load_dataset(dataset_name, trial=42):
    if dataset_name == 'iris':
        iris = load_iris()
        x = iris.data
        target = iris.target
    elif dataset_name == 'breast':
        breast = load_breast_cancer()
        x = breast.data
        target = breast.target
    elif dataset_name == 'wine':
        wine = load_wine()
        x = wine.data
        target = wine.target
    elif dataset_name == 'cifar10':
        cifar10 = fetch_openml(name='CIFAR_10_small', version=1)
        x = cifar10.data.astype('float32') / 255.0
        x = x.values
        target = cifar10.target
        label_encoder = LabelEncoder()
        target = label_encoder.fit_transform(target)
    elif dataset_name == 'fashion':
        fashion_mnist = fetch_openml(name='Fashion-MNIST', version=1)
        x = fashion_mnist.data.astype('float32') / 255.0
        x = x.values
        target = fashion_mnist.target
        label_encoder = LabelEncoder()
        target = label_encoder.fit_transform(target)
    else:
        digits = load_digits()
        x = digits.data
        target = digits.target


    target = np.array(target).reshape(-1, 1)
    encoder = OneHotEncoder()
    scaller = MinMaxScaler()
    x = scaller.fit_transform(x)
    y = encoder.fit_transform(target).toarray()

    x, y = add_synthetic_data(x, y, 200, 0.1)

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.50, random_state=trial)
    return train_x, test_x, train_y, test_y

def experiment(dataset, trial):
    train_x, test_x, train_y, test_y = load_dataset(dataset, trial)
    maxgen = 300
    generator = Generator.Generator()
    kernel_value = generator.kernel_value(train_x[:, :], kernel_option=8)
    train_x = generator.neuron(kernel_value)
    classifier = Classifier.Classifier(train_x, train_y)
    model = Model.Model()

    model_w, model_l = classifier.train_classification(train_x, train_y)
    predict = model.classification(model_w, model_l, train_x, train_y)
    for i in range(maxgen):
        model_w, model_l = classifier.train_classification(train_x, train_y)
        predict = model.classification(model_w, model_l, train_x, train_y)
        acc_train = model.accuracy_rate(predict, train_y)
        # if i%10 == 0:
        #     print(acc_train)


    train_probs = model.classification_probs(model_w, model_l, train_x, train_y)
    train_y = np.argmax(train_y, axis=1)
    train_probs = train_probs[np.arange(len(train_probs)), train_y]

    test_x = generator.kernel_value(test_x[:, :], kernel_option=8)
    predict = model.classification(model_w, model_l, test_x, test_y)
    acc_test = model.accuracy_rate(predict, test_y)
    print(acc_test)


    test_probs = model.classification_probs(model_w, model_l, test_x, test_y)
    test_y = np.argmax(test_y, axis=1)
    test_probs = test_probs[np.arange(len(test_probs)), test_y]

    return np.mean(train_probs, axis=0), np.mean(test_probs, axis=0)




dataset = 'iris'
num_of_trials = 100
train_probs = np.array([])
test_probs = np.array([])
for trial in range(num_of_trials):
    print("Trial: ", trial)
    train_prob, test_prob = experiment(dataset, trial)
    train_probs = np.append(train_probs, train_prob)
    test_probs = np.append(test_probs, test_prob)

print(np.mean(train_probs, axis=0), np.mean(test_probs, axis=0))