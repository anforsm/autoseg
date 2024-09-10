local baseline = import "autoseg/user_configs/anton/baselines/unet";

baseline + {
  training+: {
    train_dataloader+: {
      num_workers: 10,
      precache_per_worker: 1,
    },
    logging+: {
      wandb: true,
    }
  },
  model+: {
    name: "UNet_convnext_3x3x3",
    #input_image_shape: [78, 402, 402],
    #input_image_shape: [79, 403, 403],
    input_image_shape: [76, 304, 304],
    #output_image_shape: [36, 84, 84],
    #output_image_shape: [51, 191, 191],
    output_image_shape: [20, 128, 128],

    #output_image_shape: [2, 3, 3],
    #output_image_shape: [56, 56, 56],
    hyperparameters+: {
      downsample_factors: [[1, 2, 2], [1, 2, 2], [1, 2, 2]],
      kernel_size_down: [
          [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
      ],
      kernel_size_up: [
          [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
      ],
      convnext_style: true,
      fmap_inc_factor: 3,
    }
  }
}
