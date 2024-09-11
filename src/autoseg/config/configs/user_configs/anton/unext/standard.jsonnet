local defaults = import "autoseg/user_configs/anton/baselines/unet";

defaults + {
  model+: {
    name: "v2_UNeXt_Standard",
    version: "Final",
    path: "checkpoints/",
    class: "UNeXt",
    #input_image_shape: [54, 150, 150],
    #output_image_shape: [26, 60, 60],
    input_image_shape: [54, 268, 268],
    output_image_shape: [26, 56, 56],
    hyperparameters: {
      in_channels: 1,
      output_shapes: [3],
      num_fmaps: 12,
      fmap_inc_factor: 5,
      stage_ratio: [3, 3, 9, 3],
      patchify: false
    }
  },
}
