import pickle
import numpy as np
import augmentation as augm
import preprocess as pp
import h5py
import os
import ex_config as cfg
import xsets

h5_outdir = '/rois_10/'
set_name = 'sets_10.pkl'

crop_params = {'shift': (0, 0, -0.05), 'prc': (0.05, 0.05, 0.05)}

hippo_roi_L = (65 - 2, 92 + 1 - 2, 58 - 2, 85 + 1 - 2, 31 - 2, 58 + 1 - 2) # (x_min, x_max, y_min, y_max, z_min, z_max)
hippo_roi_R = (30 - 2, 57 + 1 - 2, 58 - 2, 85 + 1 - 2, 31 - 2, 58 + 1 - 2) # (x_min, x_max, y_min, y_max, z_min, z_max)

hippo_roi_L_ext = (65 - 2 - 5, 92 + 1 - 2 + 5, 58 - 2 - 5, 85 + 1 - 2 + 5, 31 - 2 - 5, 58 + 1 - 2 + 5) # (x_min, x_max, y_min, y_max, z_min, z_max)
hippo_roi_R_ext = (30 - 2 - 5, 57 + 1 - 2 + 5, 58 - 2 - 5, 85 + 1 - 2 + 5, 31 - 2 - 5, 58 + 1 - 2 + 5) # (x_min, x_max, y_min, y_max, z_min, z_max)

hippo_roi_rev_L = (31 - 2, 58 + 1 - 2, 58 - 2, 85 + 1 - 2, 65 - 2, 92 + 1 - 2) # (z_min, z_max, y_min, y_max, x_min, x_max)
hippo_roi_rev_R = (31 - 2, 58 + 1 - 2, 58 - 2, 85 + 1 - 2, 30 - 2, 57 + 1 - 2) # (z_min, z_max, y_min, y_max, x_min, x_max)

hippo_roi_rev_L_ext = (31 - 2 - 5, 58 + 1 - 2 + 5, 58 - 2 - 5, 85 + 1 - 2 + 5, 65 - 2 - 5, 92 + 1 - 2 + 5) # (z_min, z_max, y_min, y_max, x_min, x_max)
hippo_roi_rev_R_ext = (31 - 2 - 5, 58 + 1 - 2 + 5, 58 - 2 - 5, 85 + 1 - 2 + 5, 30 - 2 - 5, 57 + 1 - 2 + 5) # (z_min, z_max, y_min, y_max, x_min, x_max)

use_crops = {
    hippo_roi_L: 'L',
    hippo_roi_R: 'R',
    hippo_roi_rev_L: 'rev_L',
    hippo_roi_rev_R: 'rev_R',
    hippo_roi_L_ext: 'L_ext',
    hippo_roi_R_ext: 'R_ext',
    hippo_roi_rev_L_ext: 'rev_L_ext',
    hippo_roi_rev_R_ext: 'rev_R_ext'
}

max_augm_params = augm.AugmParams(shift=(2, 2, 2))
scale = 1.0 / 256

if not os.path.exists(cfg.h5_cache_dir + h5_outdir):
    os.makedirs(cfg.h5_cache_dir + h5_outdir)

with open(cfg.sets_dir + set_name, 'rb') as f:
    train, valid, test = pickle.load(f)

preproc = lambda item, index, crop_params: pp.full_preprocess(item, cfg.adni_root, np.float32, max_augm_params, img_index=index, crop_roi_params=crop_params)

def write_set(xset, use_smri=True, use_md=False):
    f = h5py.File(cfg.h5_cache_dir + h5_outdir + xset.name + '.h5', 'w')
    size = xset.size()
    labels = f.create_dataset('labels', shape=(size,), dtype=np.float32)
    for crop in use_crops:
        print('processing crop %s\n' % use_crops[crop])
        shape = preproc(xset.items[0], 0, crop).shape
        data_smri = f.create_dataset('data/smri_' + use_crops[crop], shape = (size,) + shape, dtype=np.float32)
        data_md = f.create_dataset('data/md_' + use_crops[crop], shape = (size,) + shape, dtype=np.float32)
        for i in range(size):
            print('writing instance %d of %d' % (i+1, size))
            if use_smri:
                data_smri[i] = preproc(xset.items[i], 0, crop) * scale
            if use_md:
                data_md[i] = preproc(xset.items[i], 1, crop) * scale
    for i in range(size):
        labels[i] = cfg.get_label_code('ternary', xset.items[i].label)
    f.close()
    print('done!')

# def test_set(xset):
#     f = h5py.File(cfg.h5_cache_dir + h5_outdir + xset.name + '.h5', 'r')
#     data_smri = f['data/smri']
#     data_smri2 = f['data/smri2']
#     data_md = f['data/md']
#     labels = f['labels']
#     print(data_smri.shape)
#     for i in range(data_smri.shape[0]):
#         print(data_smri[i].shape)
#         print(data_smri2[i].shape)
#         print(data_md[i].shape)
#         print(labels[i])
#     f.close()


train_valid = xsets.unite_sets([train, valid], 'alz_train_eval')
print(train_valid.name, train_valid.size())
write_set(train_valid)
write_set(test)
train_binary_sets = xsets.split_to_binary_sets(train_valid)
test_binary_sets = xsets.split_to_binary_sets(test)
for i in train_binary_sets:
    print(i.name, i.size())
    write_set(i)
for i in test_binary_sets:
    print(i.name, i.size())
    write_set(i)

# test_set(train_valid)
# test_set(test)