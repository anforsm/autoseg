local defaults = import "autoseg/user_configs/anton/unext/standard";

defaults + {
  model+: {
    name: "v2_UNeXt_Shallow",
    path: "checkpoints/",
    class: "UNeXt",
    input_image_shape: [54, 268, 268],
    output_image_shape: [26, 56, 56],
    hyperparameters: {
      stage_ratio: [3, 3, 3, 3],
      bottleneck_fmap_inc: 4,
      downsample_factor: 3
    }
  },
}
