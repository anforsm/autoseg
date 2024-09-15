local base = import "autoseg/user_configs/anton/unetr/anisotropic.jsonnet";


base + {
  model+: {
    name+: "_run_2"
  }
}
