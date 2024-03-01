local default = import "autoseg/defaults";

default + {
  pipeline: default.pipeline + {
    #augment: [],
  },
}
