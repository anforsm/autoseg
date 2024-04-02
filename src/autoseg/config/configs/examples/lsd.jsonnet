local default = import "autoseg/defaults";

{}
 + default

 + {
  pipeline+: {
    _order+: ["lsd_target"],
    _outputs+: ["GT_LSDS", "LSDS_WEIGHTS", "GT_LSDS_MASK"],

    lsd_target: [
      {add_local_shape_descriptor: {
        segmentation: "LABELS",
        descriptor: "GT_LSDS",
        labels_mask: "LABELS_MASK",
        lsds_mask: "LSDS_WEIGHTS",
        sigma: 40,
        downsample: 2,
      }},
    ]
  },

  model+: {
    hyperparameters+: {
      output_shapes+: [10]
    }
  },

  training+: {
    batch_outputs+: ["gt_lsds", "lsds_weights", "lsds_mask"],
    model_outputs+: ["lsds"],
    loss+: {
      _inputs+: ["lsds", "gt_lsds", "lsds_weights"],
    },
    logging+: {
      log_images+: ["gt_lsds", "lsds"]
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
