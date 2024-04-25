local defaults = import "autoseg/defaults";
local pad = import "autoseg/defaults/pad";
local zarrsource = import "autoseg/defaults/zarrsource";

defaults + {

  training+: {
    update_steps: 200000,
    log_snapshot_every: 100,
    save_every: 25000,
    val_log: 2500,
    num_val_samples: 100,
    logging+: {
      wandb: false,
    }
  },
  pipeline+: {
    source: [
      [
        [zarrsource.zarr_source("SynapseWeb/kh2015/apical", "s1")] + pad,
      ],
      {random_provider: {}}
    ],
  },
  predict+: {
    predict_with_every_n_checkpoint: 1,
    shape_increase: [0, 405, 405],
    #shape_increase: [0, 0, 0],
    source: [
      {
        path: "SynapseWeb/kh2015/oblique",
        dataset: "raw/s1",
      }
    ],
  },
  model: {
    name: "Baseline_UNet",
    path: "checkpoints/",
    #hf_path: "anforsm/" + self.name,
    hf_path: null,
    class: "UNet",
    input_image_shape: [84, 268, 268],
    output_image_shape: [56, 56, 56],
    hyperparameters: {
      in_channels: 1,
      output_shapes: [3],
      num_fmaps: 12,
      num_fmaps_out: 12,
      fmap_inc_factor: 5,
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
}
