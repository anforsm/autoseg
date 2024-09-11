local defaults = import "autoseg/user_configs/anton/baselines/unetr";
local pad = import "autoseg/defaults/pad";
local zarrsource = import "autoseg/defaults/zarrsource";

defaults + {
  model+: {
    local img_shape = [48, 272, 272],
    input_image_shape: img_shape,
    output_image_shape: img_shape,
    name: "v2_UNETR_base",
    path: "checkpoints/",
    class: "UNETR",
    hyperparameters+: {
      img_shape: img_shape,
    }
  },
}
