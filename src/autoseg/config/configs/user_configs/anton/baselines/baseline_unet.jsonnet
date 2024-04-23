local defaults = import "autoseg/defaults";

defaults + {
  # USE s1 SCALE
  model: {
    name: "Baseline_UNet",
    path: "checkpoints/",
    #hf_path: "anforsm/" + self.name,
    hf_path: null,
    class: "UNet",
    input_image_shape: [36, 268, 268],
    output_image_shape: [8, 56, 56],
    hyperparameters: {
      in_channels: 1,
      output_shapes: [3],
      num_fmaps: 12,
      num_fmaps_out: 12,
      fmap_inc_factor: 5,
      downsample_factors: [[1, 3, 3], [1, 3, 3], [3, 3, 3]],
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
    }
  },
}
