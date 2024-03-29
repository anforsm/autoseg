{
  training: {
    multi_gpu: false,
    // definition for what variables each training batch yields
    batch_outputs: ["raw", "labels", "gt_affs", "affs_weights", "gt_lsds", "lsds_weights"],
    // definition for what variables the model yields
    model_outputs: ["lsds", "affs"],
    // definition for what variables the model expects
    model_inputs: ["raw"],
    update_steps: 10000,
    log_snapshot_every: 10,
    save_every: 1000, # save model every 1000 iterations
    val_log: 10000,
    loss: {
      weighted_m_s_e_loss_double: {},
      _inputs: ["lsds", "gt_lsds", "lsds_weights", "affs", "gt_affs", "affs_weights"],
    },
    train_dataloader: {
      batch_size: 2,
      parallel: false,
      num_workers: 4,
      precache_per_worker: 4,
      input_image_shape: [36, 212, 212],
      output_image_shape: [12, 120, 120],
    },
    logging: {
      log_images: ["raw", "labels", "gt_affs", "gt_lsds", "affs", "lsds"],
      wandb: true,
    }
  }
}
