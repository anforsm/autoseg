local base = import "autoseg/user_configs/anton/baselines/unet.jsonnet";

base + {
  model+: {
    name+: "_run_3"
  }
}
