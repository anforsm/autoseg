local defaults = import "autoseg/user_configs/anton/baselines/unet";

local modifications = {
  model+: {
    name: "v2_UNet_large",
    path: "checkpoints/",
    class: "UNet",
    input_image_shape: [54, 268, 268],
    output_image_shape: [26, 56, 56],
    hyperparameters: {
      in_channels: 1,
      output_shapes: [3],
      num_fmaps: 64,
      num_fmaps_out: 64,
      fmap_inc_factor: 3,
      constant_upsample: false,
      downsample_factors: [[1, 3, 3], [1, 3, 3], [1, 3, 3]],
      kernel_size_down: [
          [[3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3]],
      ],
      kernel_size_up: [
          [[3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3]],
      ],
    }
  },

};

defaults + modifications# + dentate_prediction
