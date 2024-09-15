local base = import "autoseg/user_configs/anton/unext/patchify.jsonnet";

base + {
  model+: {
    name+: "_run_1"
  }
}
