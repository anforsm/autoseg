{
  training: {
    multi_gpu: true,
    train_dataloader: {
      batch_size: 2,
      parallel: true,
      num_workers: 4,
      precache_per_worker: 4
    },
    logging: {
      wandb: false,
    }
  }
}
