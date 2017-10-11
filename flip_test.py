import ex_config as cfg
import pickle
import preprocess as pp
import copy
import numpy as np

with open(cfg.sets_dir + 'sets_10.pkl', 'rb') as sets_file:
    train, valid, test, test_ext = pickle.load(sets_file)

for i in range(10):
    i0 = train.items[i]
    i0.preprocess.crop = pp.CropParams(global_crop=None, roi_coords=pp.CropParams.hippo_roi_L_ext,
                                       axis_flip=(False, False, False))
    i1 = copy.deepcopy(i0)
    i1.preprocess.crop = pp.CropParams(global_crop=None, roi_coords=pp.CropParams.hippo_roi_R_ext,
                                       axis_flip=(True, False, False))
    i2 = copy.deepcopy(i0)
    i2.preprocess.crop = pp.CropParams(global_crop=None, roi_coords=pp.CropParams.hippo_roi_R_ext,
                                       axis_flip=(False, True, False))
    i3 = copy.deepcopy(i0)
    i3.preprocess.crop = pp.CropParams(global_crop=None, roi_coords=pp.CropParams.hippo_roi_R_ext,
                                       axis_flip=(False, False, True))

    p0 = pp.apply_preprocess(i0, cfg.adni_root, img_index=0)
    p1 = pp.apply_preprocess(i1, cfg.adni_root, img_index=0)
    p2 = pp.apply_preprocess(i2, cfg.adni_root, img_index=0)
    p3 = pp.apply_preprocess(i3, cfg.adni_root, img_index=0)

    mse1 = np.mean((p0-p1)**2)
    mse2 = np.mean((p0-p2)**2)
    mse3 = np.mean((p0-p3)**2)

    print(mse1, mse2, mse3)
