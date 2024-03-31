import os
import random
from math import ceil
from typing import List, Tuple
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

def generate_no_drift(exp: int, magnitude: float) -> str:
    random.seed(3071980 + exp)
    n_attributes = get_drift_n_attributes()
    n_values_per_attribute = get_drift_n_attributes_values()
    total_n_instances_before_drift = get_total_n_instances_before_drift()
    total_n_instances_during_drift = get_total_n_instances_during_drift()
    total_n_instances_after_drift = get_total_n_instances_after_drift()
    num_data = total_n_instances_before_drift + total_n_instances_during_drift + total_n_instances_after_drift
    
    X, y = make_classification(n_samples=num_data, n_features=n_attributes, n_informative=n_attributes, n_redundant=0, n_clusters_per_class=1, random_state=3071980 + exp)
    X, y = shuffle(X, y, random_state=3071980 + exp)
    
    header = f"@relation 'no_drift'\n\n"
    for i in range(n_attributes):
        header += f"@attribute x{i} {{ "
        for j in range(n_values_per_attribute):
            if j == n_values_per_attribute - 1:
                header += f"{j}"
            else:
                header += f"{j}, "
        header += " }\n"
    header += "\n@data\n\n"
    
    data = ""
    for i in range(num_data):
        instance = ",".join([str(x) for x in X[i]])
        data += f"{instance}\n"
    
    return header + data

def generate_drift_gradual_bayesian(exp: int, magnitude: float) -> str:
    random.seed(3071980 + exp)
    n_attributes = get_drift_n_attributes()
    n_values_per_attribute = get_drift_n_attributes_values()
    total_n_instances_before_drift = get_total_n_instances_before_drift()
    total_n_instances_during_drift = get_total_n_instances_during_drift()
    total_n_instances_after_drift = get_total_n_instances_after_drift()
    num_data = total_n_instances_before_drift + total_n_instances_during_drift + total_n_instances_after_drift
    
    X, y = make_classification(n_samples=num_data, n_features=n_attributes, n_informative=n_attributes, n_redundant=0, n_clusters_per_class=1, random_state=3071980 + exp)
    X, y = shuffle(X, y, random_state=3071980 + exp)
    
    header = f"@relation 'drift_gradual_bayesian'\n\n"
    for i in range(n_attributes):
        header += f"@attribute x{i} {{ "
        for j in range(n_values_per_attribute):
            if j == n_values_per_attribute - 1:
                header += f"{j}"
            else:
                header += f"{j}, "
        header += " }\n"
    header += "\n@data\n\n"
    
    data = ""
    for i in range(num_data):
        instance = ",".join([str(x) for x in X[i]])
        data += f"{instance}\n"
    
    return header + data

def generate_drift_gradual(exp: int, magnitude: float) -> str:
    random.seed(3071980 + exp)
    n_attributes = get_drift_n_attributes()
    n_values_per_attribute = get_drift_n_attributes_values()
    total_n_instances_before_drift = get_total_n_instances_before_drift()
    total_n_instances_during_drift = get_total_n_instances_during_drift()
    total_n_instances_after_drift = get_total_n_instances_after_drift()
    num_data = total_n_instances_before_drift + total_n_instances_during_drift + total_n_instances_after_drift
    
    X, y = make_classification(n_samples=num_data, n_features=n_attributes, n_informative=n_attributes, n_redundant=0, n_clusters_per_class=1, random_state=3071980 + exp)
    X, y = shuffle(X, y, random_state=3071980 + exp)
    
    header = f"@relation 'drift_gradual'\n\n"
    for i in range(n_attributes):
        header += f"@attribute x{i} {{ "
        for j in range(n_values_per_attribute):
            if j == n_values_per_attribute - 1:
                header += f"{j}"
            else:
                header += f"{j}, "
        header += " }\n"
    header += "\n@data\n\n"
    
    data = ""
    for i in range(num_data):
        instance = ",".join([str(x) for x in X[i]])
        data += f"{instance}\n"
    
    return header + data

def generate_drift_gradual_swapping_generator(exp: int, magnitude: float) -> str:
    random.seed(3071980 + exp)
    n_attributes = get_drift_n_attributes()
    n_values_per_attribute = get_drift_n_attributes_values()
    total_n_instances_before_drift = get_total_n_instances_before_drift()
    total_n_instances_during_drift = get_total_n_instances_during_drift()
    total_n_instances_after_drift = get_total_n_instances_after_drift()
    num_data = total_n_instances_before_drift + total_n_instances_during_drift + total_n_instances_after_drift
    
    X, y = make_classification(n_samples=num_data, n_features=n_attributes, n_informative=n_attributes, n_redundant=0, n_clusters_per_class=1, random_state=3071980 + exp)
    X, y = shuffle(X, y, random_state=3071980 + exp)
    
    header = f"@relation 'drift_gradual_swapping_generator'\n\n"
    for i in range(n_attributes):
        header += f"@attribute x{i} {{ "
        for j in range(n_values_per_attribute):
            if j == n_values_per_attribute - 1:
                header += f"{j}"
            else:
                header += f"{j}, "
        header += " }\n"
    header += "\n@data\n\n"
    
    data = ""
    for i in range(num_data):
        instance = ",".join([str(x) for x in X[i]])
        data += f"{instance}\n"
    
    return header + data

def generate_drift_data(exp: int, magnitude: float) -> str:
    random.seed(exp)
    n_attributes = get_drift_n_attributes()
    n_values_per_attribute = get_drift_n_attributes_values()
    total_n_instances_before_drift = get_total_n_instances_before_drift()
    total_n_instances_during_drift = get_total_n_instances_during_drift()
    total_n_instances_after_drift = get_total_n_instances_after_drift()
    num_data = total_n_instances_before_drift + total_n_instances_during_drift + total_n_instances_after_drift
    
    X, y = make_classification(n_samples=num_data, n_features=n_attributes, n_informative=n_attributes, n_redundant=0, n_clusters_per_class=1, random_state=exp)
    X, y = shuffle(X, y, random_state=exp)
    
    header = f"@relation 'drift_data'\n\n"
    for i in range(n_attributes):
        header += f"@attribute x{i} {{ "
        for j in range(n_values_per_attribute):
            if j == n_values_per_attribute - 1:
                header += f"{j}"
            else:
                header += f"{j}, "
        header += " }\n"
    header += "\n@data\n\n"
    
    data = ""
    for i in range(num_data):
        instance = ",".join([str(x) for x in X[i]])
        data += f"{instance}\n"
    
    return header + data

def generate_tan_drift(exp: int, frequency: float) -> str:
    random.seed(3071980 + exp)
    n_attributes = get_drift_n_attributes()
    n_values_per_attribute = get_drift_n_attributes_values()
    total_n_instances_before_drift = get_total_n_instances_before_drift()
    total_n_instances_during_drift = get_total_n_instances_during_drift()
    total_n_instances_after_drift = get_total_n_instances_after_drift()
    num_data = total_n_instances_before_drift + total_n_instances_during_drift + total_n_instances_after_drift
    
    X, y = make_classification(n_samples=num_data, n_features=n_attributes, n_informative=n_attributes, n_redundant=0, n_clusters_per_class=1, random_state=3071980 + exp)
    X, y = shuffle(X, y, random_state=3071980 + exp)
    
    header = f"@relation 'tan_drift'\n\n"
    for i in range(n_attributes):
        header += f"@attribute x{i} {{ "
        for j in range(n_values_per_attribute):
            if j == n_values_per_attribute - 1:
                header += f"{j}"
            else:
                header += f"{j}, "
        header += " }\n"
    header += "\n@data\n\n"
    
    data = ""
    for i in range(num_data):
        instance = ",".join([str(x) for x in X[i]])
        data += f"{instance}\n"
    
    return header + data

def generate_kdb_drift(exp: int, frequency: float) -> str:
    random.seed(3071980 + exp)
    n_attributes = get_drift_n_attributes()
    n_values_per_attribute = get_drift_n_attributes_values()
    total_n_instances_before_drift = get_total_n_instances_before_drift()
    total_n_instances_during_drift = get_total_n_instances_during_drift()
    total_n_instances_after_drift = get_total_n_instances_after_drift()
    num_data = total_n_instances_before_drift + total_n_instances_during_drift + total_n_instances_after_drift
    
    X, y = make_classification(n_samples=num_data, n_features=n_attributes, n_informative=n_attributes, n_redundant=0, n_clusters_per_class=1, random_state=3071980 + exp)
    X, y = shuffle(X, y, random_state=3071980 + exp)
    
    header = f"@relation 'kdb_drift'\n\n"
    for i in range(n_attributes):
        header += f"@attribute x{i} {{ "
        for j in range(n_values_per_attribute):
            if j == n_values_per_attribute - 1:
                header += f"{j}"
            else:
                header += f"{j}, "
        header += " }\n"
    header += "\n@data\n\n"
    
    data = ""
    for i in range(num_data):
        instance = ",".join([str(x) for x in X[i]])
        data += f"{instance}\n"
    
    return header + data
def generate_simple_drift(exp, frequency):
    random.seed(3071980 + exp)
    np.random.seed(3071980 + exp)
    
    n_attributes = get_drift_n_attributes()
    n_values_per_attribute = get_drift_n_attributes_values()
    drift_length = get_total_n_instances_during_drift()
    
    X, y = make_classification(n_samples=drift_length, n_features=n_attributes, n_informative=n_attributes//2, n_redundant=n_attributes//2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    header = ""
    header += "@relation 'drift'\n\n"
    for i in range(n_attributes):
        header += "@attribute x" + str(i) + " { "
        for j in range(n_values_per_attribute):
            if j == n_values_per_attribute - 1:
                header += str(j)
            else:
                header += str(j) + ", "
        header += " }\n"
    header += "\n@data\n\n"
    
    data = ""
    for i in range(drift_length):
        for j in range(n_attributes):
            data += str(X_train[i][j]) + ","
        data += str(y_train[i]) + "\n"
    
    file_path = os.path.join(get_temp_directory(), "trainCV.arff")
    with open(file_path, "w") as f:
        f.write(header)
        f.write(data)
    
    print("Stream written to ARFF file " + file_path)
    return file_path

def generate_simple_drift_from_data(exp, num_cycles):
    source_file = get_source_file()
    source_file = randomize_training_file(source_file)
    structure = set_structure()
    
    header = ""
    header += "@relation 'contrieved'\n\n"
    for i in range(structure.numAttributes()):
        header += "@attribute x" + str(i) + " { "
        for j in range(structure.attribute(i).numValues()):
            if j == structure.attribute(i).numValues() - 1:
                header += structure.attribute(i).value(j)
            else:
                header += structure.attribute(i).value(j) + ", "
        header += " }\n"
    header += "\n@data\n\n"
    
    data = ""
    for r in range(num_cycles):
        with open(source_file, "r") as f:
            reader = f.readlines()
            for line in reader:
                data += line
    
    file_path = os.path.join(get_temp_directory(), "trainCV.arff")
    with open(file_path, "w") as f:
        f.write(header)
        f.write(data)
    
    print("Stream written to ARFF file " + file_path)
    return file_path


