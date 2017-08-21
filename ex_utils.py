import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pickle

def save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def get_nii_from_folder(folder):
    res = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.nii'):
                res.append(os.path.join(root, file))
    if len(res) > 1:
        print('WARNING. Folder %s contains more than one files' % folder)
    return res

def nii_to_array(nii_filename, data_type, fix_nan=True):
    img = nib.load(nii_filename)
    np_data = img.get_data().astype(data_type)
    if fix_nan:
        np_data = np.nan_to_num(np_data)
    return np_data

def crop_border(data, crop_prc, shift_prc):
    dims = np.array(data.shape).astype(np.float)
    pads = np.round(dims * np.array(crop_prc).astype(np.float)).astype(np.int)
    shifts = np.round(dims * np.array(shift_prc).astype(np.float)).astype(np.int)
    if pads.size != 3:
        raise NameError('unsupported number of dimensions')
    else:
        x, y, z = data.shape
        pad_x, pad_y, pad_z = pads
        sh_x, sh_y, sh_z = shifts
        data_new = data[sh_x+pad_x:x+sh_x-pad_x, sh_y+pad_y:y+sh_y-pad_y, sh_z+pad_z:z+sh_z-pad_z]
        print('cropping data:', data.shape, '->', data_new.shape)
        return data_new    

def debug_plot_median_slices(np_data, print_slices=False):
    x, y, z = np_data.shape
    slc = np_data[:, :, z//2]
    if print_slices: print(slc)
    plt.matshow(slc, interpolation='nearest', cmap='gray')
    plt.show()
    slc = np_data[:, y//2, :]
    if print_slices: print(slc)
    plt.matshow(slc, interpolation='nearest', cmap='gray')
    plt.show()
    slc = np_data[x//2, :, :]
    if print_slices: print(slc)
    plt.matshow(slc, interpolation='nearest', cmap='gray')
    plt.show()