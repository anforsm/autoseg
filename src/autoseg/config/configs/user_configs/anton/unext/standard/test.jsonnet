local base = import "autoseg/user_configs/anton/unext/standard.jsonnet";

base + {
  training+: {
    update_steps: 50000,
    log_snapshot_every: 100,
    save_every: 1000, # save model every x iterations
    overwrite_checkpoints: false,
    val_log: 1000,
    num_val_samples: 10,
  },
  model+: {
    name+: "_test"
  }
}
