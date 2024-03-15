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

def load_dataset(dataset_name):
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

    #x, y = add_synthetic_data(x, y, 100, 0.1)

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20, random_state=42)
    return train_x, test_x, train_y, test_y

def split_dataset(x, y, num_participants):
    random_asigment = np.random.randint(0, num_participants, size=len(x))
    participants_training_data = {}
    for idx in range(num_participants):
        participant_data = np.where(random_asigment == idx)[0]
        x_participant = x[participant_data]
        y_participant = y[participant_data]
        participants_training_data[idx] = (x_participant, y_participant)

    return participants_training_data

def train_model_for_participant(training_data, model_w=None, model_b=None):
    train_x, train_y = training_data
    maxgen = 100
    generator = Generator.Generator()
    kernel_value = generator.kernel_value(train_x[:, :], kernel_option=1)
    train_x = generator.neuron(kernel_value)
    classifier = Classifier.Classifier(train_x, train_y, model_w, model_b)
    model = Model.Model()
    for i in range(maxgen):
        model_w, model_l = classifier.train_classification(train_x, train_y)
        predict = model.classification(model_w, model_l, train_x, train_y)
    return model_w, model_l

def master_train(dataset):
    train_x, test_x, train_y, test_y = load_dataset(dataset)
    maxgen = 80
    generator = Generator.Generator()
    kernel_value = generator.kernel_value(train_x[:, :], kernel_option=1)
    train_x = generator.neuron(kernel_value)
    classifier = Classifier.Classifier(train_x, train_y)
    model = Model.Model()

    model_w, model_l = classifier.train_classification(train_x, train_y)
    for i in range(maxgen):
        model_w, model_l = classifier.train_classification(train_x, train_y)

    test_x = generator.kernel_value(test_x[:, :], kernel_option=1)
    predict = model.classification(model_w, model_l, test_x, test_y)
    accuracy = accuracy_score(test_y, predict)
    precision = precision_score(test_y, predict, average='weighted')
    recall = recall_score(test_y, predict, average='weighted')
    f1 = f1_score(test_y, predict, average='weighted')
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")


def experiment_trial(dataset, num_of_participants, num_of_rounds):
    train_x, test_x, train_y, test_y = load_dataset(dataset)
    participants_training_data = split_dataset(train_x, train_y, num_of_participants)

    model_w_master = None
    model_l_master = None
    for round in range(num_of_rounds):
        models_w = []
        models_l = []

        for idx in range(num_of_participants):
            model_w, model_l = train_model_for_participant(participants_training_data[idx], model_w_master, model_l_master)
            models_w.append(model_w)
            models_l.append(model_l)

        model_w_master = np.mean(np.array(models_w), axis=0)
        model_l_master = np.mean(np.array(models_l), axis=0)


    generator = Generator.Generator()
    test_x = generator.kernel_value(test_x[:, :], kernel_option=1)

    model = Model.Model()
    predict = model.classification(model_w_master, model_l_master, test_x, test_y)
    accuracy = accuracy_score(test_y, predict)
    precision = precision_score(test_y, predict, average='weighted')
    recall = recall_score(test_y, predict, average='weighted')
    f1 = f1_score(test_y, predict, average='weighted')

    return accuracy, precision, recall, f1




def experiment(dataset, num_of_participants, num_of_rounds, num_of_trials):
    acc = []
    prec = []
    rec = []
    f1_score = []
    for idx in range(num_of_trials):
        accuracy, precision, recall, f1 = experiment_trial(dataset, num_of_participants, num_of_rounds)
        acc.append(accuracy)
        prec.append(precision)
        rec.append(recall)
        f1_score.append(f1)

    # print(f"Accuracy: {np.mean(acc):.4f}")
    # print(f"Precision: {np.mean(prec):.4f}")
    # print(f"Recall: {np.mean(rec):.4f}")
    # print(f"F1 Score: {np.mean(f1_score):.4f}")
    return np.mean(acc)


def experiment_num_of_round():
    fig, axes = plt.subplots(2, 2, figsize=(15, 6))
    datasets = ['digits', 'breast', 'iris', 'wine']
    for idx in range(len(datasets)):
        dataset = datasets[idx]
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        accuracies = []
        num_rounds = np.arange(1, 5)
        for num_of_rounds in num_rounds:
            print(dataset, num_of_rounds)
            acc = experiment(dataset, 3, num_of_rounds, 10)
            accuracies.append(acc)

        ax.set_xticks(num_rounds)
        ax.plot(num_rounds, accuracies)
        ax.set_title(dataset)
    plt.tight_layout()
    plt.show()

def experiment_accuracy_num_participants():
    datasets = ['digits', 'breast', 'iris', 'wine']
    for idx in range(len(datasets)):
        dataset = datasets[idx]
        accuracies = []
        num_participants = [1, 2, 5, 8]
        for num_of_participants in num_participants:
            print(dataset, num_of_participants)
            acc = experiment(dataset, num_of_participants, 1, 10)
            accuracies.append(acc)

        print(accuracies)

experiment_accuracy_num_participants()