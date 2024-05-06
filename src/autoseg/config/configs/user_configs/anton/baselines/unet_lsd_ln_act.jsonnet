local baseline = import "autoseg/user_configs/anton/baselines/unet_lsd";

baseline + {

  model+: {
    name: "UNet_LSD_GELU_LayerNorm",
    hyperparameters+: {
      #convnext_style: true,
      activation: "GELU",
      normalization: "LayerNorm"
    }
  }
}
