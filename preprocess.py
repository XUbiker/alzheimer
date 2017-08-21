import ex_utils
import augmentation as augm

def full_preprocess(set_item, adni_root, data_type, max_augm_params, img_index = 0, crop_params=None, crop_roi_params=None):
    nii = ex_utils.get_nii_from_folder(adni_root + set_item.image_dirs[img_index])[0]
    array = ex_utils.nii_to_array(nii, data_type)
    if crop_params != None:
        array = ex_utils.crop_border(array, crop_prc=crop_params['prc'], shift_prc=crop_params['shift'])
    augmented = augm.augment(array, set_item.augm_params, max_augm_params)
    if crop_roi_params != None:
        crp = crop_roi_params # (min_x, max_x, min_y, max_y, min_z, max_z)
        augmented = augmented[crp[0]:crp[1], crp[2]:crp[3], crp[4]:crp[5]]
    return augmented
