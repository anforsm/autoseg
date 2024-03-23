{
  model: {
    name: "3D Model",
    path: "out",
    hf_path: "anforsm/3DEM",
    hyperparameters: {
      in_channels: 1,
      output_shapes: [10, 3],
      fmap_inc_factor: 5,
      downsample_factors: [[1, 2, 2], [1, 2, 2], [2, 2, 2]],
      kernel_size_down: [
          [[3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3]],
          [[1, 3, 3], [1, 3, 3]],
      ],
      kernel_size_up: [
          [[3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3]],
          [[3, 3, 3], [3, 3, 3]],
      ],
    }
  }
}
