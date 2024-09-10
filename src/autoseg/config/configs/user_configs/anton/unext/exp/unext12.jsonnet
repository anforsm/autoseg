local defaults = import "autoseg/user_configs/anton/baselines/unet";
local pad = import "autoseg/defaults/pad";
local zarrsource = import "autoseg/defaults/zarrsource";

defaults + {
  model+: {
    name: "v2_UNeXt_11.0",
    path: "checkpoints/",
    #hf_path: "anforsm/" + self.name,
    hf_path: null,
    class: "UNeXt",
    version: "11.0",
    #input_image_shape: [54, 402, 402],
    input_image_shape: [54, 268, 268],
    output_image_shape: [26, 60, 60],
    #output_image_shape: [42, 191, 191],
    hyperparameters: {
      in_channels: 1,
      output_shapes: [3],
      num_fmaps: 12,
    }
  },
}
