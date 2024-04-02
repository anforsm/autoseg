local default = import "autoseg/default";

{}
 + default
 + {
  training+: {
    multi_gpu: true
  }
}
