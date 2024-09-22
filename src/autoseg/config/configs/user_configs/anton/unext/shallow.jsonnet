local defaults = import "autoseg/user_configs/anton/unext/standard";

defaults + {
  predict+: {
    datasets: [
      {
        name: "Oblique",
        shape_increase: [-12, 405, 405],
        mask: {
          path: "SynapseWeb/kh2015/oblique",
          dataset: "labels_mask/s1"
        },
        source: {
          path: "SynapseWeb/kh2015/oblique",
          dataset: "raw/s1"
        },
        output: [{
          path: "oblique_prediction.zarr",
          dataset: "preds/affs",
          num_channels: 3,
        }]
      },
    ],
  },
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
