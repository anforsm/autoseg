local default = import "autoseg/examples/lsd";
#local default = import "autoseg/defaults";

{}
 + default
 + {model: {
    name: "UNETR_LSD",
    #name: "UNETR",
    path: "checkpoints/",
    hf_path: null,
    class: "UNETR",
    input_image_shape: [48, 208, 208],
    output_image_shape: [48, 208, 208],
    hyperparameters: {
      in_channels: 1,
      output_shapes: [3, 10],
    }
  }}
 + {
  training+: {
    logging+: {
      wandb: true
    }
  }
 }
