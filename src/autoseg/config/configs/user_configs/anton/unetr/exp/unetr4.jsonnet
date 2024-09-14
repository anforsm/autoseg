local defaults = import "autoseg/user_configs/anton/baselines/unetr";
local pad = import "autoseg/defaults/pad";
local zarrsource = import "autoseg/defaults/zarrsource";

defaults + {
  model+: {
    #local img_shape = [128, 128, 128],
    local img_shape = [128, 128, 128],
    name: "v2_UNETR_no_pad",
    input_image_shape: img_shape,
    output_image_shape: [36, 36, 36],
    hyperparameters+: {
      img_shape: img_shape,
      pad_decoder: false,
    }
  },
}
