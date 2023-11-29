import numpy as np
import os
import pickle


class Data:
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label


def prepare_data(data):
    labels = np.unique(np.array([dat.label for dat in data]))
    class_dim = len(labels)
    features = [[] for i in range(class_dim)]
    for dat in data:
        features[dat.label].append(dat.feature)
    return labels, features


def train(train_data):
    # Return
    # mean ... list with one entry per class
    #          each entry is the mean of the feature vectors of a class
    # covariance ... list with one entry per class
    #          each entry is the covariance of the feature vectors of a class
    labels, features = prepare_data(train_data)
    mean = []
    covariance = []
    for i in range(len(features)):
        feature_matrix = np.array(features[i])
        mean.append(np.mean(feature_matrix, axis=0))
        covariance.append(np.cov(feature_matrix, rowvar=False))
    return mean, covariance


def trainIdentity(train_data):
    labels, features = prepare_data(train_data)
    mean = []
    covariance = []
    for i in range(len(features)):
        feature_matrix = np.array(features[i])
        mean.append(np.mean(feature_matrix, axis=0))
        # change to identity matrix
        covariance.append(np.identity(feature_matrix.shape[1]))
    return mean, covariance


def evaluateCost(feature_vector, m, c):
    # Input
    # feature_vector ... feature vector under test
    # m     mean of the feature vectors for a class
    # c     covariance of the feature vectors of a class
    # Output
    #   some scalar proportional to the logarithm fo the probability d_j(feature_vector)
    diff = feature_vector - m
    return -0.5 * np.log(np.linalg.det(c)) - 0.5 * np.dot(np.dot(diff.T, np.linalg.inv(c)), diff)


def classify(test_data, mean, covariance):
    decisions = []
    for data in test_data:
        costs = [evaluateCost(data.feature, mean[i], covariance[i])
                 for i in range(len(mean))]
        decisions.append(np.argmax(costs))
    return decisions


def computeConfusionMatrix(decisions, test_data):
    class_labels = set([d.label for d in test_data])
    matrix = np.zeros((len(class_labels), len(class_labels)), dtype=int)
    for i, decision in enumerate(decisions):
        true_label = test_data[i].label
        matrix[true_label][decision] += 1
    return matrix


def main():
    dir = os.path.join(os.getcwd() + '/Lab10/')
    train_data = pickle.load(open(dir + "train_data.pkl", "rb"))
    test_data = pickle.load(open(dir + "test_data.pkl", "rb"))

    # Train: Compute mean and covariance for each object class from {0,1,2,3}
    # returns one list entry per object class
    mean, covariance = train(train_data)

    # for missclassification in task e)
    # mean, covariance = trainIdentity(train_data)

    # Decide: Compute decision for each feature vector from test_data
    # return a list of class indices from the set {0,1,2,3}
    decisions = classify(test_data, mean, covariance)
    print(decisions)

    # Copmute the confusion matrix
    confusion_matrix = computeConfusionMatrix(decisions, test_data)
    # [
    # [5 0 0 0]
    # [0 5 0 0]
    # [0 0 5 0]
    # [0 0 0 5]
    # ]
    # perfect classification
    print(confusion_matrix)


if __name__ == "__main__":
    main()
