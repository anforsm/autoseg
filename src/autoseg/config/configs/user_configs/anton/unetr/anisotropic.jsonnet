local defaults = import "autoseg/user_configs/anton/baselines/unetr";
local pad = import "autoseg/defaults/pad";
local zarrsource = import "autoseg/defaults/zarrsource";

defaults + {
  model+: {
    #local img_shape = [48, 240, 240],
    local img_shape = [24, 320, 320],
    name: "v3_UNETR_patch_size",
    input_image_shape: img_shape,
    output_image_shape: img_shape,
    hyperparameters+: {
      img_shape: img_shape,
      patch_size: [3, 40, 40],
      upsample_factors: [
        [3, 5, 5],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
      ]
    }
  },
}
