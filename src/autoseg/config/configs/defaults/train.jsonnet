{
  training: {
    multi_gpu: false,
    train_dataloader: {
      batch_size: 2,
      parallel: false,
      num_workers: 4,
      precache_per_worker: 4,
      input_image_shape: [36, 212, 212],
      output_image_shape: [12, 120, 120],
    },
    logging: {
      wandb: false,
    }
  }
}
