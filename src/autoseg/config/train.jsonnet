{
  training: {
    multigpu: true,
    batch_size: 8,
    training_dataloader: {
      parallel: true,
      num_workers: 4,
      precache_per_worker: 4
    }
  }
}
