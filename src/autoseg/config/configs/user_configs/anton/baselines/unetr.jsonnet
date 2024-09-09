local defaults = import "autoseg/user_configs/anton/baselines/unet";
local pad = import "autoseg/defaults/pad";
local zarrsource = import "autoseg/defaults/zarrsource";

defaults + {
  model+: {
    local img_shape = [48, 272, 272],
    name: "v2_UNETR",
    path: "checkpoints/",
    #hf_path: "anforsm/" + self.name,
    hf_path: null,
    class: "UNETR",
    input_image_shape: img_shape,
    output_image_shape: img_shape,
    hyperparameters: {
      img_shape: img_shape,
      in_channels: 1,
      #output_shapes: [3, 10],
      output_shapes: [3],
      patch_size: 16,
      num_fmaps: 12,
      embed_dim: 768,
      num_heads: 12,
    }
  },
  predict+: {
    datasets: [
      {
        name: "Oblique",
        shape_increase: [0, 0, 0],
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
}
