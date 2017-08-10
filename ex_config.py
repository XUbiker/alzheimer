caffe_folder = '/home/xubiker/dev/caffe_modified/'

h5_cache_dir = 'c:/dev/alzheimer/h5_cache/'

_codes = {
    'ternary': { 'NC': 0, 'MCI': 1, 'AD': 2 },
    'AD-MCI': { 'MCI': 0, 'AD': 1 },
    'MCI-NC': { 'NC': 0, 'MCI': 1 },
    'AD-NC': { 'NC': 0, 'AD': 1 },
}

def get_bin_label_families(label=None):
    return {
        None:  ('AD-MCI', 'MCI-NC', 'AD-NC'),
        'AD':  ('AD-MCI', 'AD-NC'),
        'MCI': ('AD-MCI', 'MCI-NC'),
        'NC':  ('AD-NC', 'MCI-NC')
    }[label]

def get_label_code(label_family, label):
    return _codes[label_family][label]

def load_caffe(caffe_root=caffe_folder):
    import sys
    pcr = caffe_root + "/python"
    if not pcr in sys.path:
        sys.path.append(pcr)

    import caffe
    print('caffe', caffe.__version__, 'loaded')  