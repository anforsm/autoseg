


{
  pipeline: {
    local pad = import "autoseg/defaults/pad",
    local zarrsource = import "autoseg/defaults/zarrsource",

    _order: ["source", "normalize", "augment", "lsd_target", "target", "post"],
    _outputs: ["RAW", "LABELS", "GT_AFFS", "AFFS_WEIGHTS", "GT_AFFS_MASK", "LABELS_MASK"],

    source: [
      [
        [zarrsource.zarr_source("SynapseWeb/kh2015/oblique")] + pad,
        [zarrsource.zarr_source("SynapseWeb/kh2015/spine")] + pad,
      ],
      {random_provider: {}}
    ],

    normalize: [
      {normalize: {
        array: "RAW"
      }},
    ],

    augment: [
      {elastic_augment: {
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
      {smooth_array: {
        array: "RAW"
      }},
      {grow_boundary: {
        labels: "LABELS",
        only_xy: true,
      }}
    ],

    target: [
      {add_affinities: {
        labels: "LABELS",
        labels_mask: "LABELS_MASK",
        affinities: "GT_AFFS",
        affinities_mask: "GT_AFFS_MASK",
        affinity_neighborhood: [
          [-1,0,0],
          [0,-1,0],
          [0,0,-1]
        ]
      }},
      {balance_labels: {
        labels: "GT_AFFS",
        scales: "AFFS_WEIGHTS",
        mask: "GT_AFFS_MASK",
      }},
    ],

    post: [
      {intensity_scale_shift: {
        array: "RAW",
        scale: 2,
        shift: -1
      }},
      {unsqueeze: {
        arrays: ["RAW"],
        axis: 0,
      }}
    ]
  }
}
