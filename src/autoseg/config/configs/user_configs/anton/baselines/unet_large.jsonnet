# This enhanced UNet uses Local Shape Descriptors, Batch Normalization and GeLU Activation.
local baseline = import "autoseg/user_configs/anton/baselines/unet";

baseline
 + {
  model+: {
    name: "v2_UNet_Large",
    input_image_shape: [54, 268, 268],
    output_image_shape: [18, 84, 84],
    hyperparameters+: {
      in_channels: 1,
      output_shapes: [3],
      num_fmaps: 24,
      num_fmaps_out: 24,
      fmap_inc_factor: 3,
      constant_upsample: false,
      downsample_factors: [[1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2]],
      kernel_size_down: [
          [[3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3]],
      ],
      kernel_size_up: [
          [[3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3]],
      ],
    }
  },
}
