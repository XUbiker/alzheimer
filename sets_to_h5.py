import pickle
import numpy as np
import augmentation as augm
import preprocess as pp
import h5py
import os
import ex_config as cfg

if not os.path.exists(cfg.h5_cache_dir):
    os.makedirs(cfg.h5_cache_dir)

adni_root = 'C:/dev/ADNI_Multimodal/dataset/'
sets_path = 'sets.pkl'
crop_params = {'shift': (0, 0, -0.05), 'prc': (0.05, 0.05, 0.05)}
max_augm_params = augm.AugmParams(shift=(2, 2, 2))

label_dict = {'NC': 0, 'MCI': 1, 'AD': 2}

with open(sets_path, 'rb') as f:
    train, valid, test = pickle.load(f)

preprocess = lambda item: pp.full_preprocess(item, adni_root, np.float32, max_augm_params, img_index=0, crop_params=crop_params)

def write_set(xset):
    f = h5py.File(cfg.h5_cache_dir + xset.name + '.h5', 'w')
    shape = preprocess(xset.items[0]).shape
    size = xset.size()
    data = f.create_dataset('data/smri', shape = (size,) + shape, dtype=np.float32)
    labels = f.create_dataset('labels', shape = (size,), dtype=np.float32)
    for i in range(size):
        print('writing instance %d of %d' % (i+1, size))
        data[i] = preprocess(xset.items[i])
        labels[i] = cfg.get_label_code('ternary', xset.items[i].label)
    f.close()

def test_set(xset):
    f = h5py.File(cfg.h5_cache_dir + xset.name + '.h5', 'r')
    data = f['data/smri']
    labels = f['labels']
    print(data.shape)
    for i in range(data.shape[0]):
        print(data[i].shape)
        print(labels[i])
    f.close()


# write_set(test)
# test_set(train)s
