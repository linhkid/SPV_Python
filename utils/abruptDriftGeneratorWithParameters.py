import random
from typing import List

class Attribute:
    def __init__(self, name: str, values: List[str]):
        self.name = name
        self.values = values

class DenseInstance:
    def __init__(self, num_attributes: int):
        self.dataset = None
        self.values = [None] * num_attributes
        self.class_value = None

    def set_dataset(self, dataset):
        self.dataset = dataset

    def set_value(self, attribute_index: int, value):
        self.values[attribute_index] = value

    def set_class_value(self, class_value):
        self.class_value = class_value

class Instances:
    def __init__(self, name: str, attributes: List[Attribute], class_index: int):
        self.name = name
        self.attributes = attributes
        self.class_index = class_index

class InstancesHeader:
    def __init__(self, instances: Instances):
        self.instances = instances

    def num_attributes(self):
        return len(self.instances.attributes)

class FastVector:
    def __init__(self, attributes: List[Attribute]):
        self.attributes = attributes

class InstanceExample:
    def __init__(self, instance: DenseInstance):
        self.instance = instance

class TaskMonitor:
    pass

class ObjectRepository:
    pass

class DriftGenerator:
    def estimated_remaining_instances(self):
        return -1

    def has_more_instances(self):
        return True

    def is_restartable(self):
        return True

    def restart(self):
        self.n_instances_generated_so_far = 0

    def get_description(self, sb, indent):
        pass

    def get_purpose_string(self):
        return "Generates a stream with an abrupt drift of given magnitude."

    def get_header(self):
        return self.stream_header

    def generate_header(self):
        attributes = self.get_header_attributes(self.n_attributes, self.n_values_per_attribute)
        self.stream_header = InstancesHeader(Instances(self.get_cli_creation_string(InstanceStream), attributes, 0))
        self.stream_header.set_class_index(self.stream_header.num_attributes() - 1)

    def next_instance(self):
        px = self.pxbd if self.n_instances_generated_so_far < self.burn_in_n_instances else self.pxad
        pygx = self.pygxbd if self.n_instances_generated_so_far < self.burn_in_n_instances else self.pygxad
        inst = DenseInstance(self.stream_header.num_attributes())
        inst.set_dataset(self.stream_header)
        indexes = [0] * self.n_attributes

        for a in range(self.n_attributes):
            rand = random.uniform(0.0, 1.0)
            chosen_val = 0
            sum_proba = px[a][chosen_val]
            while not isclose(rand, sum_proba):
                chosen_val += 1
                sum_proba += px[a][chosen_val]
            indexes[a] = chosen_val
            inst.set_value(a, chosen_val)

        line_no_cpt = self.get_index(*indexes)
        rand = random.uniform(0.0, 1.0)
        chosen_class_value = 0
        sum_proba = pygx[line_no_cpt][chosen_class_value]
        while not isclose(rand, sum_proba):
            chosen_class_value += 1
            sum_proba += pygx[line_no_cpt][chosen_class_value]
        inst.set_class_value(chosen_class_value)
        self.n_instances_generated_so_far += 1

        return InstanceExample(inst)

    def prepare_for_use_impl(self, monitor, repository):
        self.generate_header()
        random.seed(self.seed)
        self.r = random.Random()
        self.r.seed(self.seed)
        self.n_instances_generated_so_far = 0

    def get_index(self, *indexes):
        index = indexes[0]
        for i in range(1, len(indexes)):
            index *= self.n_values_per_attribute
            index += indexes[i]
        return index

    def set_prio_dist_before_drift(self, p):
        self.pxbd = p

    def set_prio_dist_after_drift(self, p):
        self.pxad = p

    def set_cond_dist_after_drift(self, p):
        self.pygxad = p

    def set_cond_dist_before_drift(self, p):
        self.pygxbd = p

class AbruptDriftGeneratorWithParameters(DriftGenerator):
    def __init__(self):
        self.stream_header = None
        self.pxbd = None
        self.pygxbd = None
        self.pxad = None
        self.pygxad = None
        self.r = None
        self.n_instances_generated_so_far = 0

    def estimated_remaining_instances(self):
        return -1

    def has_more_instances(self):
        return True

    def is_restartable(self):
        return True

    def restart(self):
        self.n_instances_generated_so_far = 0

    def get_description(self, sb, indent):
        pass

    def get_purpose_string(self):
        return "Generates a stream with an abrupt drift of given magnitude."

    def get_header(self):
        return self.stream_header

    def generate_header(self):
        attributes = self.get_header_attributes(self.n_attributes, self.n_values_per_attribute)
        self.stream_header = InstancesHeader(Instances(self.get_cli_creation_string(InstanceStream), attributes, 0))
        self.stream_header.set_class_index(self.stream_header.num_attributes() - 1)

    def next_instance(self):
        px = self.pxbd if self.n_instances_generated_so_far < self.burn_in_n_instances else self.pxad
        pygx = self.pygxbd if self.n_instances_generated_so_far < self.burn_in_n_instances else self.pygxad
        inst = DenseInstance(self.stream_header.num_attributes())
        inst.set_dataset(self.stream_header)
        indexes = [0] * self.n_attributes

        for a in range(self.n_attributes):
            rand = self.r.uniform(0.0, 1.0)
            chosen_val = 0
            sum_proba = px[a][chosen_val]
            while not isclose(rand, sum_proba):
                chosen_val += 1
                sum_proba += px[a][chosen_val]
            indexes[a] = chosen_val
            inst.set_value(a, chosen_val)

        line_no_cpt = self.get_index(*indexes)
        rand = self.r.uniform(0.0, 1.0)
        chosen_class_value = 0
        sum_proba = pygx[line_no_cpt][chosen_class_value]
        while not isclose(rand, sum_proba):
            chosen_class_value += 1
            sum_proba += pygx[line_no_cpt][chosen_class_value]
        inst.set_class_value(chosen_class_value)
        self.n_instances_generated_so_far += 1

        return InstanceExample(inst)

    def prepare_for_use_impl(self, monitor, repository):
        self.generate_header()
        random.seed(self.seed)
        self.r = random.Random()
        self.r.seed(self.seed)
        self.n_instances_generated_so_far = 0

    def get_index(self, *indexes):
        index = indexes[0]
        for i in range(1, len(indexes)):
            index *= self.n_values_per_attribute
            index += indexes[i]
        return index

    def set_prio_dist_before_drift(self, p):
        self.pxbd = p

    def set_prio_dist_after_drift(self, p):
        self.pxad = p

    def set_cond_dist_after_drift(self, p):
        self.pygxad = p

    def set_cond_dist_before_drift(self, p):
        self.pygxbd = p


