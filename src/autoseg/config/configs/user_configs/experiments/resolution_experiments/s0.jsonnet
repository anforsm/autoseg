local default = import "autoseg/user_configs/experiments/resolution_experiments/default";

default + {
  model: {
    name: "UNet scale 0"
  }
}
