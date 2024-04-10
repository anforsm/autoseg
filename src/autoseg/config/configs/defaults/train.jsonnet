local minimal_pipeline = import "autoseg/defaults/minimal_pipeline";

{training:
  {
    multi_gpu: false,
    // definition for what variables each training batch yields
    batch_outputs: ["raw", "labels", "gt_affs", "affs_weights", "affs_mask", "labels_mask"],
    // definition for what variables the model yields
    model_outputs: ["affs"],
    // definition for what variables the model expects
    model_inputs: ["raw"],
    update_steps: 200000,
    log_snapshot_every: 1000,
    save_every: 10000, # save model every x iterations
    overwrite_checkpoints: false,
    val_log: 1000,
    num_val_samples: 100,
    save_best: true,
    learning_rate: 5e-5,
    loss: {
      weighted_m_s_e_loss: {},
      _inputs: ["affs", "gt_affs", "affs_weights"],
    },
    train_dataloader: {
      batch_size: 1,
      parallel: true,
      num_workers: 20,
      precache_per_worker: 2,
      use_gunpowder_precache: true,
      # FOR UNET
      #input_image_shape: [48, 196, 196],
      #output_image_shape: [16, 104, 104],
      # FOR UNETR
      input_image_shape: [48, 208, 208],
      output_image_shape: [48, 208, 208],
    },
    logging: {
      log_images: ["raw", "labels", "gt_affs", "affs"],
      wandb: false,
    },
    val_dataloader: self.train_dataloader + minimal_pipeline + {
      # can't use 2 dataloaders with gunpowder pipeines?
      parallel: false,
      use_gunpowder_precache: false,
    }
  }
}
