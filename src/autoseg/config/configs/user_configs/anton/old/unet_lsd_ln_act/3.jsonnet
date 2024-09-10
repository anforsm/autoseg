local base = import "autoseg/user_configs/anton/baselines/unet_lsd_ln_act.jsonnet";

base + {
  model+: {
    name+: "_run_3"
  }
}
