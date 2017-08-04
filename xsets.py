class XSetItem:
    def __init__(self, label, image_dirs, augm_params):
        self.label = label
        self.image_dirs = image_dirs
        self.augm_params = augm_params
    def __str__(self):
        return '::' + self.label + ', ' + str(self.image_dirs) + ', ' + str(self.augm_params) + '::'
        
class XSet:
    def __init__(self, name='', items = None):
        self.name = name
        self.items = [] if items == None else items
    def size(self):
        return len(self.items)
    def add(self, item):
        self.items.append(item)
    def add_all(self, other_set):
        self.items.extend(other_set.items)
    def shuffle(self):
        import numpy.random as rnd
        rnd.shuffle(self.items)
    def print(self):
        print('%s (%d instances):' % (self.name, len(self.items)))
        for i in self.items: print(i)
