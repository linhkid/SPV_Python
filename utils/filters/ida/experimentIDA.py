import decimal
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import KBinsDiscretizer

datasets = ["airlines-n.arff", "elecNormNew-n.arff", "gas-sensor-n.arff", "powersupply-n.arff", "sensor-n.arff"]

def classify_discretized(dataset):
    learner = SGDClassifier(loss="hinge", alpha=0.001, learning_rate="constant", eta0=0.01)
    stream = np.genfromtxt("/home/forestier/Dropbox/#decay/datasets/clean/" + dataset, delimiter=",")
    X = stream[:, :-1]
    y = stream[:, -1]
    filter = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
    X_discretized = filter.fit_transform(X)
    learner.partial_fit(X_discretized, y, classes=np.unique(y))
    accuracy = 1 - learner.score(X_discretized, y)
    return accuracy

def classify_original(dataset):
    learner = SGDClassifier(loss="hinge", alpha=0.001, learning_rate="constant", eta0=0.01)
    stream = np.genfromtxt("/home/forestier/Dropbox/#decay/datasets/clean/" + dataset, delimiter=",")
    X = stream[:, :-1]
    y = stream[:, -1]
    learner.partial_fit(X, y, classes=np.unique(y))
    accuracy = 1 - learner.score(X, y)
    return accuracy

for dataset in datasets:
    print(dataset[:dataset.rfind(".")], decimal.Decimal(classify_original(dataset)).quantize(decimal.Decimal("0.####")), decimal.Decimal(classify_discretized(dataset)).quantize(decimal.Decimal("0.####")))


