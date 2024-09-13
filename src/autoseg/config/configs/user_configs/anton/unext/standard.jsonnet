local defaults = import "autoseg/user_configs/anton/baselines/unet";

defaults + {
  predict+: {
    datasets: [
      {
        name: "Oblique",
        shape_increase: [-12, 0, 0],
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
    name: "v2_UNeXt_Standard",
    version: "Final",
    path: "checkpoints/",
    class: "UNeXt",
    input_image_shape: [54, 192, 192],
    output_image_shape: [26, 100, 100],
    #input_image_shape: [54, 268, 268],
    #output_image_shape: [26, 56, 56],
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
