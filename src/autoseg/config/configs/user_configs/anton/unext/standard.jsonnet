local defaults = import "autoseg/user_configs/anton/baselines/unet";

defaults + {
  model+: {
    name: "v3_UNeXt_Standard",
    version: "Final",
    path: "checkpoints/",
    class: "UNeXt",
    # v2 has input_image shape = (54, 192, 192)
    # v2 has output image shape = (26, 100, 100)
    input_image_shape: [54, 268, 268],
    output_image_shape: [26, 56, 56],
    #input_image_shape: [54, 268, 268],
    #output_image_shape: [26, 56, 56],
    hyperparameters: {
      in_channels: 1,
      output_shapes: [3],
      num_fmaps: 12,
      fmap_inc_factor: 5,
      stage_ratio: [3, 3, 9, 3],
      # v2 has downsample_factor=3
      downsample_factor: 3,
      patchify: false
    }
  },
}
