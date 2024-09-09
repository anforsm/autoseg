# This enhanced UNet uses Local Shape Descriptors, Batch Normalization and GeLU Activation.
local baseline = import "autoseg/user_configs/anton/baselines/unet";
local dentate_prediction = import "autoseg/user_configs/anton/utils/dentate_predict_lsd";

local lsd_target = {lsd_target: [
  {add_local_shape_descriptor: {
    segmentation: "LABELS",
    descriptor: "GT_LSDS",
    labels_mask: "LABELS_MASK",
    lsds_mask: "LSDS_WEIGHTS",
    sigma: 80,
    downsample: 1,
  }},
]};


baseline
 + {
  pipeline+: {
    _order+: ["lsd_target"],
    _outputs+: ["GT_LSDS", "LSDS_WEIGHTS"],
  } + lsd_target,

  model+: {
    name: "v2_UNet_LSD",
    hyperparameters+: {
      output_shapes+: [10]
    }
  },

  training+: {
    batch_outputs+: ["gt_lsds", "lsds_weights"],
    model_outputs+: ["lsds"],
    loss+: {
      _inputs+: ["lsds", "gt_lsds", "lsds_weights"],
    },
    logging+: {
      log_images+: ["gt_lsds", "lsds"],
      wandb: true
    },
    train_dataloader+: {
      num_workers: 40,
    },
    val_dataloader+: {
      pipeline+: {
        _order+: ["lsd_target"],
        _outputs+: ["GT_LSDS", "LSDS_WEIGHTS"],
      } + lsd_target
    }

  },

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
        },
        {
          path: "oblique_prediction.zarr",
          dataset: "preds/lsds",
          num_channels: 10,
        }]
      },
    ],

  }
}
