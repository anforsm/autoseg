local base = import "autoseg/user_configs/anton/baselines/unetr.jsonnet";

base + {
  model+: {
    name+: "_run_3"
  }
}
