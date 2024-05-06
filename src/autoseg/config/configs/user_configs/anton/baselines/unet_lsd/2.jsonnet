local base = import "autoseg/user_configs/anton/baselines/unet_lsd.jsonnet";

base + {
  model+: {
    name+: "_run_2"
  }
}
