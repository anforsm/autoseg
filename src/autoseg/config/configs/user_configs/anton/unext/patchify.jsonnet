local defaults = import "autoseg/user_configs/anton/unext/standard";

defaults + {
  model+: {
    name: "v2_UNeXt_Patchify",
    #input_image_shape: [54, 150, 150],
    #output_image_shape: [26, 60, 60],
    hyperparameters+: {
      patchify: true
    }
  },
}