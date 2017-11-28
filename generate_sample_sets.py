import numpy.random as rnd
import random
import os
from xsets import XSet, XSetItem
from preprocess import AugmParams, PreprocessParams
import ex_config as cfg
import pickle
import ex_utils


def generate_augm_set(dirs_with_labels, new_size, max_augm_params):

    xset = XSet()
    
    if new_size is None or len(dirs_with_labels) == new_size:
        for d in dirs_with_labels:
            xset.add(XSetItem(label=d[0], image_dirs=(d[1], d[2])))
        return xset
    
    augm_coeff = new_size // len(dirs_with_labels)

    for d in dirs_with_labels:
        xset.add(XSetItem(label=d[0], image_dirs=(d[1], d[2])))
        for _ in range(augm_coeff-1):
            ap = AugmParams.random(max_augm_params.shift, max_augm_params.sigma)
            xset.add(XSetItem(label=d[0], image_dirs=(d[1], d[2]), preprocess=PreprocessParams(augmentation=ap)))
    for _ in range(new_size - len(dirs_with_labels) * augm_coeff):
        d = random.choice(dirs_with_labels)
        ap = AugmParams.random(max_augm_params.shift, max_augm_params.sigma)
        xset.add(XSetItem(label=d[0], image_dirs=(d[1], d[2]), preprocess=PreprocessParams(augmentation=ap)))
    return xset


def generate_samples_from_adni2(adni_root, max_augm_params, augm_factor, prefix_name='alz', test_prc=0.25,
                                shuffle_data=True, debug=True):
    
    stage_dirs = {
        'AD': '/AD/',
        'MCI': '/MCI/',
        'NC': '/NC/'
    }

    stage_dirs_root = {k: adni_root + v for k, v in stage_dirs.items()}

    class_size = {k: len(os.listdir(stage_dirs_root[k])) for k in stage_dirs_root}
    print('source patients:', class_size)

    ts = int(min(class_size.values()) * test_prc)
    test_size = {k: ts for k in stage_dirs_root}
    train_size = {k: class_size[k] - test_size[k] for k in stage_dirs_root}
    
    print('source patients used for train & validation:', train_size)
    print('source patients used for test', test_size)

    train_size_balanced = int(max(train_size.values()) * augm_factor)
    test_size_balanced = int(max(test_size.values()) * augm_factor)
    print('train & validation data will be augmented to %d samples by each class' % train_size_balanced)
    print('test data will be augmented to %d samples by each class' % ts)

    sample_sets = {
        'train': XSet(name=prefix_name + '_train'),
        'test_0': XSet(name=prefix_name + '_test_0'),
        'test_1': XSet(name=prefix_name + '_test_1'),
        'test_2': XSet(name=prefix_name + '_test_2'),
    }


    for k in stage_dirs_root:
        stage_dir = stage_dirs[k]
        patient_dirs = os.listdir(stage_dirs_root[k])
        rnd.shuffle(patient_dirs)

        test_dirs = patient_dirs[:test_size[k]]
        train_dirs = patient_dirs[test_size[k]:]
                                 
        train_lists = [(k, stage_dir + d + '/SMRI/', stage_dir + d + '/MD/') for d in train_dirs]
        test_lists = [(k, stage_dir + d + '/SMRI/', stage_dir + d + '/MD/') for d in test_dirs]

        sample_sets['train'].add_all(generate_augm_set(train_lists, train_size_balanced, max_augm_params))
        sample_sets['test_0'].add_all(generate_augm_set(test_lists, None, None))
        sample_sets['test_1'].add_all(generate_augm_set(test_lists, test_size_balanced,
                                                        AugmParams(max_augm_params.shift, sigma=0.0)))
        sample_sets['test_2'].add_all(generate_augm_set(test_lists, test_size_balanced, max_augm_params))

    if shuffle_data:
        for s in sample_sets:
            sample_sets[s].shuffle()

    if debug:
        for s in sample_sets:
            sample_sets[s].print()

    return sample_sets


def save_params(params, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(params, f)


def generate_sets(file_path, params, debug=True):
    sample_sets = generate_samples_from_adni2(
        params['adni_root'],
        params['max_augm'], test_prc=params['test_prc'],
        augm_factor=params['augm_factor'],
        prefix_name=params['prefix_name'],
        shuffle_data=False, debug=debug
    )
    ex_utils.save_pickle(sample_sets, file_path)


lists_params = {
    'adni_root': cfg.adni_root,
    'prefix_name': 'alz',
    'max_augm': AugmParams(shift=(2, 2, 2), sigma=1.2),
    'test_prc': 0.25,
    'augm_factor': 20
}

generate_sets('./sets/sets_10_2.pkl', lists_params, debug=True)
