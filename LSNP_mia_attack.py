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
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import os
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
        noise = np.random.uniform(low=-1.0, high=1.0, size=sample.shape) * noise_level
        sample = sample + noise
        x = np.vstack([x, sample])
        y = np.vstack([y, y[idx]])
    return x, y

def generate_synthetic_data(x, y, num_samples, noise_level):
    new_x = []
    new_y = []
    for idx in range(num_samples):
        sample = x[idx % len(x)]
        noise = np.random.uniform(low=0.0, high=1.0, size=sample.shape) * noise_level
        sample = sample + noise
        new_x.append(sample)
        new_y.append(y[idx % len(x)])
    return np.array(new_x), np.array(new_y)

def generate_noisy_data(x, y, noise_level):
    new_x = []
    new_y = []
    num_features = int(noise_level * len(x[0]))
    for idx in range(len(x)):
        sample = x[idx]
        random_indices = np.random.randint(low=0, high=len(sample), size=num_features)
        sample[random_indices] = np.random.uniform(low=0.0, high=1.0, size=num_features)
        new_x.append(sample)
        new_y.append(y[idx])
    return np.array(new_x), np.array(new_y)


def load_dataset(dataset_name, noise_level=0):
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

    #x, y = add_synthetic_data(x, y, 10000, noise_level)
    x, y = generate_synthetic_data(x, y, len(x), noise_level)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.50)
    return train_x, test_x, train_y, test_y

def train_lsnp(dataset, noise_level):
    train_x, test_x, train_y, test_y = load_dataset(dataset, noise_level)
    maxgen = 100
    generator = Generator.Generator()
    kernel_value = generator.kernel_value(train_x[:, :], kernel_option=1)
    train_x = generator.neuron(kernel_value)
    classifier = Classifier.Classifier(train_x, train_y)
    model = Model.Model()

    model_w, model_l = classifier.train_classification(train_x, train_y)
    for i in range(maxgen):
        model_w, model_l = classifier.train_classification(train_x, train_y)
    predict = model.classification(model_w, model_l, test_x, test_y)
    acc_test = model.accuracy_rate(predict, test_y)
    #print("Testing accuracy for noise ", noise_level, ": ", acc_test)

    probs_training = model.raw_classification(model_w, model_l, train_x, train_y)
    probs_testing = model.raw_classification(model_w, model_l, test_x, test_y)
    probs_training = np.concatenate((probs_training, np.argmax(train_y, axis=1)[:, np.newaxis]), axis=1)
    probs_testing = np.concatenate((probs_testing, np.argmax(test_y, axis=1)[:, np.newaxis]), axis=1)

    probs_data = np.concatenate((probs_training, probs_testing), axis=0)
    probs_labels = np.zeros(len(probs_data))
    probs_labels[:len(probs_training)] = 1

    return probs_data, probs_labels


def experiment(dataset, c, s):
    scaller = MinMaxScaler()

    classifier = RandomForestClassifier(n_estimators=256)
    for idx in range(s):
        shadow_probs, shadow_labels = train_lsnp(dataset, 0.2)
        indices_c = np.where(shadow_probs[:, -1] == c)
        shadow_probs = shadow_probs[:, :-1]
        shadow_probs = shadow_probs[indices_c]
        shadow_labels = shadow_labels[indices_c]

        shadow_probs = scaller.fit_transform(shadow_probs)

        classifier.fit(shadow_probs, shadow_labels)


    probs, labels = train_lsnp(dataset, 0.2)
    indices_c = np.where(probs[:, -1] == c)
    probs = probs[:, :-1]
    probs = probs[indices_c]
    labels = labels[indices_c]

    probs = scaller.fit_transform(probs)

    labels_pred = classifier.predict(probs)

    accuracy = accuracy_score(labels, labels_pred)
    precision = precision_score(labels, labels_pred, average='weighted')
    recall = recall_score(labels, labels_pred, average='weighted')
    f1 = f1_score(labels, labels_pred, average='weighted')
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    return accuracy, precision, recall, f1

#Impact of the class
def experiment_impact_of_the_class():
    dataset = 'digits'
    class_accuracies = np.array([])
    for c in range(10):
        accuracy, precision, recall, f1 = experiment(dataset, c, 50)
        class_accuracies = np.append(class_accuracies, accuracy)

    plt.bar(np.arange(10), class_accuracies)
    plt.xticks(np.arange(len(class_accuracies)), np.arange(len(class_accuracies)))
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Attack accuracy for each class')
    plt.show()

def experiment_impact_of_shadow_num():
    dataset = 'digits'
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for c in range(10):
        row = c // 5
        col = c % 5
        ax = axes[row, col]
        class_accuracies = np.array([])

        for s in [2, 6, 10, 14, 18, 22, 26, 30]:
            accuracy_trials = np.array([])
            for idx in range(1):
                accuracy, precision, recall, f1 = experiment(dataset, c, s)
                accuracy_trials = np.append(accuracy_trials, accuracy)
            class_accuracies = np.append(class_accuracies, np.mean(accuracy_trials))
            print(c, s, np.mean(accuracy_trials))
        ax.set_xticks([2, 6, 10, 14, 18, 22, 26, 30])
        ax.scatter(np.array([2, 6, 10, 14, 18, 22, 26, 30]), class_accuracies)
        ax.set_title(f'Class {c}')
    plt.tight_layout()
    plt.show()


experiment_impact_of_shadow_num()