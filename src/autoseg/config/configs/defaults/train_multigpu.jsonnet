{
  training: {
    multi_gpu: true,
    save_every: 1000, # save model every 1000 iterations
    // definition for what variables each training batch yields
    batch_outputs: ["raw", "labels", "gt_affs", "affs_weights", "gt_lsds", "lsds_weights"],
    // definition for what variables the model yields
    model_outputs: ["lsds", "affs"],
    // definition for what variables the model expects
    model_inputs: ["raw"],
    loss: {
      weighted_m_s_e_loss_double: {},
      _inputs: ["lsds", "gt_lsds", "lsds_weights", "affs", "gt_affs", "affs_weights"],
    },
    train_dataloader: {
      batch_size: 2,
      parallel: true,
      num_workers: 4,
      precache_per_worker: 4,
      input_image_shape: [36, 212, 212],
      output_image_shape: [12, 120, 120]
    },
    logging: {
      wandb: false,
    }
  }
}
