local baseline = import "autoseg/user_configs/anton/baselines/unet";

baseline + {
  training+: {
    #logging+: {
    #  wandb: false
    #},
    optimizer: {
      "AdamWScheduleFree": {
        lr: 5*5e-5,
        warmup_steps: 10000,
      }
    }
  },
  model+: {
    name: "UNet_baseline_schedule_free_2",
  }
}
