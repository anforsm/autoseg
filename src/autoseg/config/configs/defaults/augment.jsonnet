{augment: [
  {elastic_augment: {
    #control_point_spacing: [1, "$voxel_size[0]", "$voxel_size[0]"],
    control_point_spacing: [1, 50, 50],
    jitter_sigma: [0, 5, 5],
    scale_interval: [0.5, 2.0],
    rotation_interval: [0, 3.141592 / 2],
    subsample: 4
  }},
  {simple_augment: {
    transpose_only: [1, 2]
  }},
  {noise_augment: {
    array: "RAW",
  }},
  {intensity_augment: {
    array: "RAW",
    scale_min: 0.9,
    scale_max: 1.1,
    shift_min: -0.1,
    shift_max: 0.1
  }},
  {grow_boundary: {
    labels: "LABELS",
    only_xy: true,
  }}
]}
