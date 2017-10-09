import ex_utils
import numpy as np
from scipy.ndimage.filters import gaussian_filter


class AugmParams:
    def __init__(self, shift=(0, 0, 0), sigma=0.0):
        self.shift = shift
        self.sigma = sigma

    @staticmethod
    def random(max_shift=(0, 0, 0), max_sigma_blur=0.0):
        while True:
            shift_x = np.random.randint(-max_shift[0], max_shift[0])
            shift_y = np.random.randint(-max_shift[1], max_shift[1])
            shift_z = np.random.randint(-max_shift[2], max_shift[2])
            blur_sigma = float(np.random.randint(1000)) / 1000 * max_sigma_blur
            if shift_x + shift_y + shift_z + blur_sigma > 0:
                return AugmParams((shift_x, shift_y, shift_z), blur_sigma)

    def __str__(self):
        return '<' + str(self.shift) + ', ' + str(self.sigma) + '>'

    def __repr__(self):
        return str(self)


class CropParams:
    # ROI sizes (x_min, x_max, y_min, y_max, z_min, z_max):
    hippo_roi_L = (65, 92, 58, 85, 31, 58)
    hippo_roi_R = (30, 57, 58, 85, 31, 58)
    ext_s = 5  # additional size of extended roi in each direction
    hippo_roi_L_ext = (65 - ext_s, 92 + ext_s, 58 - ext_s, 85 + ext_s, 31 - ext_s, 58 + ext_s)
    hippo_roi_R_ext = (30 - ext_s, 57 + ext_s, 58 - ext_s, 85 + ext_s, 31 - ext_s, 58 + ext_s)

    def __init__(self, global_crop=None, roi_coords=None, axis_flip=(False, False, False)):
        self.global_crop = global_crop
        self.roi_coords = roi_coords
        self.axis_flip = axis_flip

    def __str__(self):
        return '<' + str(self.global_crop) + ', ' + str(self.roi_coords) + ', ' + str(self.axis_flip) + '>'

    def __repr__(self):
        return str(self)


class PreprocessParams:
    def __init__(self, augmentation=AugmParams(), crop=CropParams()):
        self.augmentation = augmentation
        self.crop = crop

    def __str__(self):
        return '<' + str(self.augmentation) + ', ' + str(self.crop) + '>'

    def __repr__(self):
        return str(self)


def apply_preprocess(set_item, root_folder, img_index, data_type=np.float32):
    nii = ex_utils.get_nii_from_folder(root_folder + set_item.image_dirs[img_index])[0]
    data = ex_utils.nii_to_array(nii, data_type)
    pp = set_item.preprocess
    # if pp.global_crop is not None:
    #     data = ex_utils.crop_border(data, crop_prc=pp.global_crop['prc'], shift_prc=pp.global_crop['shift'])
    data = data if pp.augmentation.sigma == 0 else gaussian_filter(data, sigma=pp.augmentation.sigma)
    if pp.crop.roi_coords is not None:
        crd = pp.crop.roi_coords
        sh = pp.augmentation.shift
        # crop (min_x + sh_x : max_x + sh_x, min_y + sh_y : max_y + sh_y, min_z + sh_z : max_z + sh_z)
        data = data[crd[0] + sh[0]:crd[1] + sh[0], crd[2] + sh[1]:crd[3] + sh[1], crd[4] + sh[2]:crd[5] + sh[2]]
    for i in range(3):
        if pp.crop.axis_flip[i]:
            data = np.flip(data, i)
    return data
