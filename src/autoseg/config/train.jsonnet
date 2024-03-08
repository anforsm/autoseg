{
  training: {
    multi_gpu: false,
    train_dataloader: {
      batch_size: 2,
      parallel: false,
      num_workers: 4,
      precache_per_worker: 4
    },
    logging: {
      wandb: false,
    }
  }
}
