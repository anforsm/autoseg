local default = import "autoseg/user_configs/anton/baselines/unet";
#local default = import "autoseg/examples/lsd";
#local model = import "autoseg/user_configs/anton/models/unetr";
#local model = import "autoseg/examples/lsd";

{}
 + default
 #+ model.get_model_config()
 + {
  training+: {
  },
  predict+: {
    shape_increase: [0, 800, 800]
  },
  model+: {
    input_image_shape: [84, 268, 268],
    output_image_shape: [56, 56, 56],
  }
 }
