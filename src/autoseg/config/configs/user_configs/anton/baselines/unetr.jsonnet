local defaults = import "autoseg/user_configs/anton/baselines/unet";
local pad = import "autoseg/defaults/pad";
local zarrsource = import "autoseg/defaults/zarrsource";

defaults + {
  pipeline+: {
    source: [
      [
        [zarrsource.zarr_source("SynapseWeb/kh2015/apical", "s1")] + pad,
        #[zarrsource.zarr_source("SynapseWeb/kh2015/oblique", "s1")] + pad,
        #[zarrsource.zarr_source("SynapseWeb/kh2015/spine", "s1")] + pad,
      ],
      {random_provider: {}},
    ],
  },
  model+: {
    local img_shape = [48, 272, 272],
    name: "v2_UNETR",
    path: "checkpoints/",
    #hf_path: "anforsm/" + self.name,
    hf_path: null,
    class: "UNETR",
    input_image_shape: img_shape,
    output_image_shape: img_shape,
    hyperparameters: {
      img_shape: img_shape,
      in_channels: 1,
      #output_shapes: [3, 10],
      output_shapes: [3],
      patch_size: 16,
      num_fmaps: 12,
      embed_dim: 768,
      num_heads: 12,
    }
  },
  training+: {
    #logging+: {
    #  wandb: false
    #},
    train_dataloader+: {
      num_workers: 40,
    }
  },
}
