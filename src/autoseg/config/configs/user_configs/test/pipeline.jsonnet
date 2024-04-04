


{
  pipeline: {
    local pad = import "autoseg/defaults/pad",
    local zarrsource = import "autoseg/defaults/zarrsource",

    _order: ["source", "normalize", "augment", "target", "post"],
    _outputs: ["RAW", "LABELS", "GT_AFFS", "AFFS_WEIGHTS", "GT_AFFS_MASK", "LABELS_MASK", "GT_LSDS", "LSDS_WEIGHTS"],

    source: [
        [zarrsource.zarr_source("SynapseWeb/kh2015/apical")] + pad,
    ],

    normalize: [
      {normalize: {
        array: "RAW"
      }},
    ],

    augment: [
      {elastic_augment: {
        control_point_spacing: [1, 50, 50],
        jitter_sigma: [0, 1, 1],
        scale_interval: [1.0, 3.0],
        rotation_interval: [0, 3.141592 / 2],
        subsample: 4,
        prob_slip: 0.01,
        prob_shift: 0.01,
        max_misalign: 1,
      }},
      {simple_augment: {
        transpose_only: [1, 2]
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
    ],
    target: [
      {add_local_shape_descriptor: {
        segmentation: "LABELS",
        descriptor: "GT_LSDS",
        labels_mask: "LABELS_MASK",
        lsds_mask: "LSDS_WEIGHTS",
        sigma: 80,
        downsample: 2,
      }},
      {grow_boundary: {
        labels: "LABELS",
        only_xy: true,
      }},
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
