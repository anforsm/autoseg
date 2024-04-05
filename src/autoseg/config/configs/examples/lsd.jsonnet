local default = import "autoseg/defaults";

local lsd_target = {lsd_target: [
  {add_local_shape_descriptor: {
    segmentation: "LABELS",
    descriptor: "GT_LSDS",
    labels_mask: "LABELS_MASK",
    lsds_mask: "LSDS_WEIGHTS",
    sigma: 80,
    downsample: 2,
  }},
]};

# Returns
{}
 + default
 + {
  pipeline+: {
    _order+: ["lsd_target"],
    _outputs+: ["GT_LSDS", "LSDS_WEIGHTS"],
  } + lsd_target,

  model+: {
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
      log_images+: ["gt_lsds", "lsds"]
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
