local baseline = import "autoseg/user_configs/anton/baselines/unet";
local pad = import "autoseg/defaults/pad";
local zarrsource = import "autoseg/defaults/zarrsource";

baseline + {
  training+: {
    train_dataloader+: {
      batch_size: 32,
    }
  },

  model+: {
    name: "UNet_BS",
  },

  pipeline+: {
    source: [
      [
        [zarrsource.zarr_source("SynapseWeb/kh2015/apical", "s1")] + pad,
      ],
      {random_provider: {}}
    ],
  },
}
