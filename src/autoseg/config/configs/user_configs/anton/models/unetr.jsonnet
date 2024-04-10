{get_model_config(model_name="3DEM_TESTING") ::
  {model: {
    name: "UNETR",
    path: "checkpoints/" + self.name + "/",
    #hf_path: "anforsm/" + self.name,
    hf_path: null,
    class: "UNETR",
    hyperparameters: {
      in_channels: 1,
      output_shapes: [3, 10],
      #fmap_inc_factor: 5,
      #downsample_factors: [[1, 2, 2], [1, 2, 2], [2, 2, 2]],
      #kernel_size_down: [
      #    [[3, 3, 3], [3, 3, 3]],
      #    [[3, 3, 3], [3, 3, 3]],
      #    [[3, 3, 3], [3, 3, 3]],
      #    [[3, 3, 3], [3, 3, 3]],
      #],
      #kernel_size_up: [
      #    [[3, 3, 3], [3, 3, 3]],
      #    [[3, 3, 3], [3, 3, 3]],
      #    [[3, 3, 3], [3, 3, 3]],
      #],
      #activation: "GELU",
      #normalization: "BatchNorm",
    }
  }},
}
