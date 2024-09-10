local base = import "autoseg/user_configs/anton/baselines/unet_bs.jsonnet";

base + {
  model+: {
    name+: "_run_3"
  }
}
