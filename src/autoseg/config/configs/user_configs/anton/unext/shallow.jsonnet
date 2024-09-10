local defaults = import "autoseg/user_configs/anton/unext/standard";

defaults + {
  model+: {
    name: "v2_UNeXt_Shallow",
    path: "checkpoints/",
    class: "UNeXt",
    hyperparameters: {
      stage_ratio: [3, 3, 3, 3],
      bottleneck_fmap_inc: 4,
      downsample_factor: 3
    }
  },
}
