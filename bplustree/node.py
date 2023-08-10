import random

class Node:
    def __init__(self, dataset, order, reference_point):
        self.order, self.numKeys = order, 0
        self.keys = []  # will only contain indices to original data
        self.pointers = []  # will contain pointer to child nodes
        self.key_distances = []
        self.parent, self.next_key = None, None
        self.check_leaf = False
        self.reference_point = reference_point
        self.node_radius = 0
        self.entropy = 0
        self.node_id = random.randint(1, 50000)
        self.dataset = dataset
        self.node_index = None
        self.unique_labels = None

    def insert_at_leaf(self, key, key_ref_dist):
        if self.keys:
            i = self.get_closest(key_ref_dist)
            # self.keys = self.keys[:i] + [key] + self.keys[i:]
            self.keys.insert(i, key)
            # self.key_distances = self.key_distances[:i] + [key_ref_dist] + self.key_distances[i:]
            self.key_distances.insert(i, key_ref_dist)
        else:
            self.keys, self.key_distances = [key], [key_ref_dist]

    def get_closest(self, key_ref_dist):
        node_keys = self.keys
        low, high = 0, len(node_keys) - 1
        lowDiff, highDiff = self.key_distances[low], self.key_distances[high]
        if key_ref_dist < lowDiff:
            return low
        if key_ref_dist > highDiff:
            return high + 1

        while low <= high:
            mid = (low + high) // 2
            midDiff = self.key_distances[mid]
            if midDiff == key_ref_dist:
                return mid + 1
            elif key_ref_dist < midDiff:
                high = mid - 1
            else:
                low = mid + 1
        lowDiff, highDiff = self.key_distances[low], self.key_distances[high]
        return low if lowDiff < highDiff else high