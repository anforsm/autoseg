local defaults = import "autoseg/user_configs/anton/baselines/unetr";
local pad = import "autoseg/defaults/pad";
local zarrsource = import "autoseg/defaults/zarrsource";

defaults + {
  model+: {
    name: "v2_UNETR_base",
    path: "checkpoints/",
    class: "UNETR",
  },
}
