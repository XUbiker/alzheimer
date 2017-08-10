import numpy.random as rnd

class AugmParams:
    def __init__(self, shift=(0, 0, 0), sigma=0.0):
        self.shift = shift
        self.sigma = sigma
    def __str__(self):
        return '::' + str(self.shift) + ', ' + str(self.sigma) + '::'
    def __repr__(self):
        return str(self)
    def trunc_random(max_augm_params):
        max_shift = max_augm_params.shift
        max_blur = max_augm_params.sigma
        while True:
            shift_x = rnd.randint(-max_shift[0], max_shift[0])
            shift_y = rnd.randint(-max_shift[1], max_shift[1])
            shift_z = rnd.randint(-max_shift[2], max_shift[2])
            blur_sigma = float(rnd.randint(1000)) / 1000 * max_blur
            if shift_x + shift_y + shift_z + blur_sigma > 0:
                return AugmParams(shift=(shift_x, shift_y, shift_z), sigma=blur_sigma)

            
def augment(data, augm_params, max_augm_params):

    from scipy.ndimage.filters import gaussian_filter
    
    blur_sigma = augm_params.sigma
    shift = augm_params.shift
    max_shift = max_augm_params.shift

    size = (
                data.shape[0] - 2 * max_shift[0],
                data.shape[1] - 2 * max_shift[1],
                data.shape[2] - 2 * max_shift[2]
           )

    blurred = data if blur_sigma == 0 else gaussian_filter(data, sigma = blur_sigma)
    sub_data = blurred[max_shift[0] + shift[0] : size[0] + max_shift[0] + shift[0],
                       max_shift[1] + shift[1] : size[1] + max_shift[1] + shift[1],
                       max_shift[2] + shift[2] : size[2] + max_shift[2] + shift[2]]
    return sub_data