local defaults = import "autoseg/user_configs/anton/baselines/unetr";
local pad = import "autoseg/defaults/pad";
local zarrsource = import "autoseg/defaults/zarrsource";

defaults + {

  pipeline+: {
    source: [
      [
        [zarrsource.zarr_source("SynapseWeb/kh2015/apical.zarr", "resampled")] + pad,
      ],
      {random_provider: {}}
    ],
  },

  training+: {
    val_dataloader+: {
      pipeline+: {
        source: [
          [
            [zarrsource.zarr_source("SynapseWeb/kh2015/spine.zarr", "resampled")] + pad,
          ],
          {random_provider: {}}
        ],
      },
    }
  },

  predict+: {
    predict_with_every_n_checkpoint: 1,
    multi_gpu: false,
    datasets: [
      {
        name: "Oblique",
        shape_increase: [-12, 405, 405],
        mask: {
          path: "SynapseWeb/kh2015/oblique.zarr",
          dataset: "labels_mask/resampled"
        },
        source: {
          path: "SynapseWeb/kh2015/oblique.zarr",
          dataset: "raw/resampled"
        },
        output: [{
          path: "oblique_prediction.zarr",
          dataset: "preds/affs/resampled",
          num_channels: 3,
        }]
      },
    ],
  },

  model+: {
    name: "v2_UNETR_isotropic_training",
  },
}
