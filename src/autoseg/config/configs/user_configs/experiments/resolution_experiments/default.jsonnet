local default = import "autoseg/examples/lsd";

{}
 + default
 + {
  training+: {
    update_steps: 200000,
    log_snapshot_every: 2500,
    save_every: 25000,
    val_log: 2500,
    num_val_samples: 100,
  }
 }
