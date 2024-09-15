local base = import "autoseg/user_configs/anton/baselines/unet_large.jsonnet";

base + {
  model+: {
    name+: "_run_3"
  }
}
