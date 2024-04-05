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
    update_steps: 10000,
    log_snapshot_every: 100,
    save_every: 1000, # save model every x iterations
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
      num_workers: 10,
      precache_per_worker: 1,
      # change shapes to
      #input_image_shape: [48, 196, 196],
      input_image_shape: [36, 212, 212],
      output_image_shape: [4, 120, 120],
      #output_image_shape: [12, 120, 120],
      #output_image_shape: [16, 104, 104],
    },
    logging: {
      log_images: ["raw", "labels", "gt_affs", "affs"],
      wandb: true,
    },
    val_dataloader: self.train_dataloader + minimal_pipeline
  }
}
