import pickle
import numpy as np
import preprocess as pp
import h5py
import os
import xsets
import copy
from joblib import Parallel, delayed
import ex_utils as utl
from configparser import ConfigParser


cfg = ConfigParser()
cfg.read('config.ini')


def h5_write_grouped_sets(xsets, h5_dir, h5_base_name=None):
    adni_dir = cfg.get('dir', 'adni_dir')
    if h5_base_name is None:
        h5_base_name = xsets[0].name
    print('writing file: %s' % h5_base_name)
    scale = 1.0 / 256
    os.makedirs(h5_dir, exist_ok=True)
    f = h5py.File(h5_dir + h5_base_name + '.h5', 'w')
    for xset in xsets:
        size = xset.size()
        shape = pp.apply_preprocess(xset.items[0], adni_dir, img_index=0).shape
        print('processing subset: %s (%s) [%d samples]' % (xset.name, xset.tag, xset.size()))
        tag = '' if xset.tag == '' else '_' + xset.tag
        data_smri = f.create_dataset('data/smri' + tag, shape=(size,) + shape, dtype=np.float32)
        data_md = f.create_dataset('data/md' + tag, shape=(size,) + shape, dtype=np.float32)
        labels = f.create_dataset('labels/labels' + tag, shape=(size,), dtype=np.float32)
        patients = f.create_dataset('patients' + tag, shape=(size,), dtype=h5py.special_dtype(vlen=str))
        for i in range(size):
            print('writing instance %d of %d' % (i + 1, size))
            data_smri[i] = pp.apply_preprocess(xset.items[i], adni_dir, img_index=0) * scale
            data_md[i] = pp.apply_preprocess(xset.items[i], adni_dir, img_index=1) * scale
            labels[i] = utl.get_label_code(label_family='ternary', label=xset.items[i].label)
            patients[i] = str(xset.items[i])
    f.close()

    # --- check resulting h5 file ---
    # f = h5py.File(h5_dir + h5_base_name + '.h5', 'r')
    # for xset in xsets:
    #     tag = '' if xset.tag == '' else '_' + xset.tag
    #     print(f['patients' + tag])
    #     for s in range(xset.size()):
    #         print(f['patients'+tag][s])
    # f.close()

    print('done!')


def h5_write(xset, h5_dir):
    h5_write_grouped_sets((xset,), h5_dir, xset.name)


def preprocess_set(xset, expand_roi_size=0):
    exp_sfx = '_e' + str(expand_roi_size)
    sets = []
    roi_L = pp.CropParams.expand_roi(pp.CropParams.hippo_roi_L, expand_roi_size)
    roi_R = pp.CropParams.expand_roi(pp.CropParams.hippo_roi_R, expand_roi_size)
    xset_L = xsets.XSet(name=xset.name+exp_sfx, tag='L')
    xset_R = xsets.XSet(name=xset.name+exp_sfx, tag='R')
    xset_LR = xsets.XSet(name=xset.name+exp_sfx, tag='LR')
    for item in xset.items:
        i1, i2, i3 = [copy.deepcopy(item) for _ in range(3)]
        i1.preprocess.crop = pp.CropParams(global_crop=None, roi_coords=roi_L, axis_flip=(False, False, False))
        i2.preprocess.crop = pp.CropParams(global_crop=None, roi_coords=roi_R, axis_flip=(True, False, False))
        i3.preprocess.crop = pp.CropParams(global_crop=None, roi_coords=roi_R, axis_flip=(False, False, False))
        xset_L.add(i1)
        xset_R.add(i3)
        xset_LR.add(i1)
        xset_LR.add(i2)
    sets.extend((xset_L, xset_R, xset_LR))
    return sets


def preprocess_and_write(h5_dir, samples, exp_size, write_ternary, write_binary, n_parallel):
    # --- Preprocess sets and write them into h5 files ---
    if __name__ == '__main__':
        all_grouped_sets = []
        for _, sample in samples.items():
            binary_sets = [[], ]
            ternary_sets = []
            for ps in preprocess_set(sample, exp_size):
                if write_ternary:
                    ternary_sets.append(ps)
                if write_binary:
                    for idx, xset in enumerate(xsets.split_to_binary_sets(ps)):
                        if idx >= len(binary_sets): binary_sets.append([])
                        binary_sets[idx].append(xset)
            if write_binary:
                all_grouped_sets.extend(binary_sets)
            if write_ternary:
                all_grouped_sets.append(ternary_sets)
        Parallel(n_jobs=n_parallel)(delayed(h5_write_grouped_sets)(i, h5_dir) for i in all_grouped_sets)


# ---------- load saved sets, preprocess them and write to h5 files ----------
with open(cfg.get('dir', 'sets_dir') + cfg.get('augment', 'sample_filename'), 'rb') as sets_file:
    sample_sets = pickle.load(sets_file)
    for exp_size in list(map(int, cfg.get('samples', 'exp_sizes').split(','))):
        preprocess_and_write(
            h5_dir=cfg.get('dir', 'h5_cache_dir') + cfg.get('augment', 'sample_filename') + '/',
            samples=sample_sets,
            exp_size=exp_size,
            write_binary=cfg.getboolean('samples', 'write_binary'),
            write_ternary=cfg.getboolean('samples', 'write_ternary'),
            n_parallel=6
        )