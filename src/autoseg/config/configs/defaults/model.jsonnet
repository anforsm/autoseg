{get_model_config(model_name="3DEM_TESTING") ::
  {model: {
    name: "UNet_extra_latest_2",
    path: "checkpoints/",
    #hf_path: "anforsm/" + self.name,
    hf_path: null,
    class: "UNet",
    hyperparameters: {
      in_channels: 1,
      output_shapes: [3],
      fmap_inc_factor: 5,
      downsample_factors: [[1, 2, 2], [1, 2, 2], [2, 2, 2]],
      kernel_size_down: [
          [[3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3]],
      ],
      kernel_size_up: [
          [[3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3]],
      ],
      #activation: "GELU",
      #normalization: "BatchNorm",
    }
  }},
}
