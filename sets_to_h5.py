import pickle
import numpy as np
import augmentation as augm
import preprocess as pp
import h5py
import os
import ex_config as cfg
import xsets


adni_root = 'C:/dev/ADNI_Multimodal/dataset/'
h5_subdir = '/rois/'
sets_path = 'sets_5.pkl'
crop_params = {'shift': (0, 0, -0.05), 'prc': (0.05, 0.05, 0.05)}
crop_roi_params = (65-2, 92+1-2, 58-2, 85+1-2, 31-2, 58+1-2) # max_shift substracted

max_augm_params = augm.AugmParams(shift=(2, 2, 2))

scale = 1.0 / 256

if not os.path.exists(cfg.h5_cache_dir + h5_subdir):
    os.makedirs(cfg.h5_cache_dir + h5_subdir)

with open(sets_path, 'rb') as f:
    train, valid, test = pickle.load(f)

preprocess = lambda item: pp.full_preprocess(item, adni_root, np.float32, max_augm_params, img_index=0, crop_roi_params=crop_roi_params)

def write_set(xset):
    f = h5py.File(cfg.h5_cache_dir + h5_subdir + xset.name + '.h5', 'w')
    shape = preprocess(xset.items[0]).shape
    size = xset.size()
    data = f.create_dataset('data/smri', shape = (size,) + shape, dtype=np.float32)
    labels = f.create_dataset('labels', shape = (size,), dtype=np.float32)
    for i in range(size):
        print('writing instance %d of %d' % (i+1, size))
        data[i] = preprocess(xset.items[i]) * scale
        labels[i] = cfg.get_label_code('ternary', xset.items[i].label)
    f.close()

def test_set(xset):
    f = h5py.File(cfg.h5_cache_dir + h5_subdir + xset.name + '.h5', 'r')
    data = f['data/smri']
    labels = f['labels']
    print(data.shape)
    for i in range(data.shape[0]):
        print(data[i].shape)
        print(labels[i])
    f.close()


train_valid = xsets.unite_sets([train, valid], 'alz_train_eval')

write_set(train_valid)
write_set(test)
test_set(train_valid)
test_set(test)
