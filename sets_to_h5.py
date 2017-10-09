import pickle
import numpy as np
import preprocess as pp
import h5py
import os
import ex_config as cfg
import xsets
import copy


def h5_write_grouped_sets(xsets, h5_dir, h5_base_name=None):
    if h5_base_name is None:
        h5_base_name = xsets[0].name
    print('writing file: %s' % h5_base_name)
    scale = 1.0 / 256
    os.makedirs(h5_dir, exist_ok=True)
    f = h5py.File(h5_dir + h5_base_name + '.h5', 'w')
    for xset in xsets:
        size = xset.size()
        shape = pp.apply_preprocess(xset.items[0], cfg.adni_root, img_index=0).shape
        print('processing subset: %s (%s)' % (xset.name, xset.tag))
        tag = '' if xset.tag == '' else '_' + xset.tag
        data_smri = f.create_dataset('data/smri' + tag, shape=(size,) + shape, dtype=np.float32)
        data_md = f.create_dataset('data/md' + tag, shape=(size,) + shape, dtype=np.float32)
        labels = f.create_dataset('labels/labels' + tag, shape=(size,), dtype=np.float32)
        for i in range(size):
            print('writing instance %d of %d' % (i + 1, size))
            data_smri[i] = pp.apply_preprocess(xset.items[i], cfg.adni_root, img_index=0) * scale
            data_md[i] = pp.apply_preprocess(xset.items[i], cfg.adni_root, img_index=1) * scale
            labels[i] = cfg.get_label_code(label_family='ternary', label=xset.items[i].label)
    f.close()
    print('done!')


def h5_write(xset, h5_dir):
    h5_write_grouped_sets((xset), h5_dir, xset.name)


def preprocess_set(xset):
    xset_L = xsets.XSet(name=xset.name, tag='L')
    xset_R = xsets.XSet(name=xset.name, tag='R')
    xset_LR = xsets.XSet(name=xset.name, tag='LR')
    for item in xset.items:
        i1, i2, i3 = [copy.deepcopy(item) for _ in range(3)]
        i1.preprocess.crop = pp.CropParams(global_crop=None, roi_coords=pp.CropParams.hippo_roi_L_ext,
                                           axis_flip=(False, False, False))
        i2.preprocess.crop = pp.CropParams(global_crop=None, roi_coords=pp.CropParams.hippo_roi_R_ext,
                                           axis_flip=(True, False, False))
        i3.preprocess.crop = pp.CropParams(global_crop=None, roi_coords=pp.CropParams.hippo_roi_R_ext,
                                           axis_flip=(False, False, False))
        xset_L.add(i1)
        xset_R.add(i3)
        xset_LR.add(i1)
        xset_LR.add(i2)
    return xset_L, xset_R, xset_LR

# --- Load saved sets ---
with open(cfg.sets_dir + 'sets_10.pkl', 'rb') as sets_file:
    train, valid, test, test_ext = pickle.load(sets_file)
train_valid = xsets.unite_sets([train, valid], 'alz_train_eval')

# --- Preprocess sets and write them into h5 files ---
h5_dir = cfg.h5_cache_dir+'/rois_10/'
for sample_set in (train_valid, test, test_ext):
    ternary_sets = []
    binary_sets = [[], ]
    for ps in preprocess_set(sample_set):
        ternary_sets.append(ps)
        for idx, xset in enumerate(xsets.split_to_binary_sets(ps)):
            if idx >= len(binary_sets): binary_sets.append([])
            binary_sets[idx].append(xset)
    h5_write_grouped_sets(ternary_sets, h5_dir)
    for i in binary_sets:
        h5_write_grouped_sets(i, h5_dir)
