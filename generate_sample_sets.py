import numpy.random as rnd
import random
import os
from xsets import XSet, XSetItem
from preprocess import AugmParams, PreprocessParams
import ex_config as cfg


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


def generate_samples_from_adni2(adni_root, max_augm_params, augm_factor, prefix_name='alz', valid_prc=0.25,
                                test_prc=0.25, shuffle_data=True, debug=True):
    
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
    valid_size = {k: int(class_size[k] * valid_prc) for k in stage_dirs_root}
    train_size = {k: class_size[k] - test_size[k] - valid_size[k] for k in stage_dirs_root}
    
    print('source patients used for train:', train_size)
    print('source patients used for validation:', valid_size)
    print('source patients used for test', test_size)

    train_size_balanced = int(max(train_size.values()) * augm_factor)
    valid_size_balanced = int(max(valid_size.values()) * augm_factor)
    test_size_balanced = int(max(test_size.values()) * augm_factor)
    print('train data will be augmented to %d samples by each class' % train_size_balanced)
    print('validation data will be augmented to %d samples by each class' % valid_size_balanced)
    print('test data will be augmented to %d samples by each class' % ts)
    
    train_set = XSet(name=prefix_name + '_train')
    valid_set = XSet(name=prefix_name + '_valid')
    test_set = XSet(name=prefix_name + '_test')
    test_ext_set = XSet(name=prefix_name + '_test_ext')

    for k in stage_dirs_root:
        stage_dir = stage_dirs[k]
        patient_dirs = os.listdir(stage_dirs_root[k])
        rnd.shuffle(patient_dirs)

        test_dirs = patient_dirs[:test_size[k]]
        valid_dirs = patient_dirs[test_size[k]:test_size[k]+valid_size[k]]
        train_dirs = patient_dirs[test_size[k]+valid_size[k]:]
                                 
        train_lists = [(k, stage_dir + d + '/SMRI/', stage_dir + d + '/MD/') for d in train_dirs]
        valid_lists = [(k, stage_dir + d + '/SMRI/', stage_dir + d + '/MD/') for d in valid_dirs]
        test_lists = [(k, stage_dir + d + '/SMRI/', stage_dir + d + '/MD/') for d in test_dirs]
        
        train_set.add_all(generate_augm_set(train_lists, train_size_balanced, max_augm_params))
        valid_set.add_all(generate_augm_set(valid_lists, valid_size_balanced, max_augm_params))
        test_set.add_all(generate_augm_set(test_lists, None, None))
        test_ext_set.add_all(generate_augm_set(test_lists, test_size_balanced, AugmParams(max_augm_params.shift, sigma=0.0)))

    if shuffle_data:
        train_set.shuffle()
        valid_set.shuffle()
        test_set.shuffle()
        test_ext_set.shuffle()

    if debug:
        train_set.print()
        valid_set.print()
        test_set.print()
        test_ext_set.print()

    return train_set, valid_set, test_set, test_ext_set


def save_params(params, file_path):
    import pickle
    with open(file_path, 'wb') as f:
        pickle.dump(params, f)


def generate_sets(lists_file_path, params, debug=True):
    import ex_utils
    train_set, valid_set, test_set, test_ext_set = generate_samples_from_adni2(
        params['adni_root'],
        params['max_augm'], test_prc=params['test_prc'], valid_prc=params['valid_prc'],
        augm_factor=params['augm_factor'],
        prefix_name=params['prefix_name'],
        shuffle_data=False, debug=debug
    )
    ex_utils.save_pickle((train_set, valid_set, test_set, test_ext_set), lists_file_path)


lists_params = {
    'adni_root': cfg.adni_root,
    'prefix_name': 'alz',
    'max_augm': AugmParams(shift=(2, 2, 2), sigma=1.2),
    'test_prc': 0.25,
    'valid_prc': 0.25,
    'augm_factor': 10
}

import ex_utils
# ex_utils.save_pickle(lists_params, 'params.pkl')
generate_sets('./sets/sets_10.pkl', lists_params, debug=True)
