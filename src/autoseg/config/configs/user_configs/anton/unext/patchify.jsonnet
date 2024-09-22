local defaults = import "autoseg/user_configs/anton/unext/standard";

defaults + {
  model+: {
    name: "v3_UNeXt_Patchify",
    #input_image_shape: [54, 150, 150],
    #output_image_shape: [26, 60, 60],
    input_image_shape: [54, 268*2, 268*2],
    output_image_shape: [26, 56, 56],

    hyperparameters+: {
      patchify: true
    }
  },
}
