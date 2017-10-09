import random as rnd
from preprocess import PreprocessParams


class XSetItem:
    def __init__(self, label, image_dirs, preprocess=PreprocessParams()):
        self.label = label
        self.image_dirs = image_dirs
        self.preprocess = preprocess

    def __str__(self):
        return '::' + self.label + ', ' + str(self.image_dirs) + ', ' + str(self.preprocess) + '::'


class XSet:
    def __init__(self, name='', tag='', items=None):
        self.name = name
        self.tag = name if tag == '' else tag
        self.items = [] if items is None else items

    def size(self):
        return len(self.items)

    def add(self, item):
        self.items.append(item)

    def add_all(self, other_set):
        self.items.extend(other_set.items)

    def shuffle(self):
        rnd.shuffle(self.items)

    def __str__(self):
        return 'set:: name={0}, tag={1}, size={2}'.format( self.name, self.tag, self.size())

    def print(self):
        print('%s (%s), %d instances:' % (self.name, self.tag, len(self.items)))
        for i in self.items:
            print(i)


def unite_sets(source_sets, united_name, united_tag=''):
    united_set = XSet(united_name, united_tag)
    for s in source_sets:
        united_set.items.extend(s.items)
    return united_set


def split_to_binary_sets(source_set):
    unique_labels = sorted(set(map(lambda item: item.label, source_set.items)))
    binary_sets = []
    for i in range(len(unique_labels) - 1):
        for j in range(i + 1, len(unique_labels)):
            print('processing splitting:', unique_labels[i], unique_labels[j])
            new_name = source_set.name + '_' + unique_labels[i] + '_' + unique_labels[j]
            new_items = [item for item in source_set.items if
                         item.label == unique_labels[i] or item.label == unique_labels[j]]
            binary_sets.append(XSet(new_name, source_set.tag, items=new_items))
    return binary_sets
