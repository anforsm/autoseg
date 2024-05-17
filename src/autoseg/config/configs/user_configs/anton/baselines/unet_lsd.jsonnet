# This enhanced UNet uses Local Shape Descriptors, Batch Normalization and GeLU Activation.
local baseline = import "autoseg/user_configs/anton/baselines/unet";

local lsd_target = {lsd_target: [
  {add_local_shape_descriptor: {
    segmentation: "LABELS",
    descriptor: "GT_LSDS",
    labels_mask: "LABELS_MASK",
    lsds_mask: "LSDS_WEIGHTS",
    sigma: 80,
    downsample: 4,
  }},
]};


baseline
 + {
  pipeline+: {
    _order+: ["lsd_target"],
    _outputs+: ["GT_LSDS", "LSDS_WEIGHTS"],
  } + lsd_target,

  model+: {
    name: "UNet_LSD_OSA",
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
    outputs+: [{
        path: "multiout.zarr",
        dataset: "preds/lsds",
        num_channels: 10,
      }]
  }
}
