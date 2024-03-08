
local trainconfig = import "autoseg/train";

{
  pipeline: {
    local padconfig = import "autoseg/pad",
    local zarrsource = import "autoseg/zarrsource",

    local raw = "RAW",
    local labels = "LABELS",
    local labels_mask = "LABELS_MASK",
    local affinities = "GT_AFFS",
    local affinities_mask = "GT_AFFS_MASK",
    local affinities_weights = "AFFS_WEIGHTS",


    _order: ["source", "normalize", "augment", "target"],

    source: [
      [
        [zarrsource.zarr_source("SynapseWeb/kh2015/oblique")] + padconfig,
        [zarrsource.zarr_source("SynapseWeb/kh2015/spine")] + padconfig,
      ],
      {random_provider: {}}
    ],

    normalize: [
      {normalize: {
        array: raw
      }},
    ],
    augment: [
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
        array: raw,
      }},
      {intensity_augment: {
        array: raw,
        scale_min: 0.9,
        scale_max: 1.1,
        shift_min: -0.1,
        shift_max: 0.1
      }},
      {grow_boundary: {
        labels: labels,
        only_xy: true,
      }}
    ],
    target: [
      {add_affinities: {
        labels: labels,
        labels_mask: labels_mask,
        affinities: affinities,
        affinities_mask: affinities_mask,
        affinity_neighborhood: [
          [-1,0,0],
          [0,-1,0],
          [0,0,-1]
        ]
      }},
      {balance_labels: {
        labels: affinities,
        scales: affinities_weights,
        mask: affinities_mask,
      }},
      {intensity_scale_shift: {
        array: raw,
        scale: 2,
        shift: -1
      }},
      {unsqueeze: {
        arrays: [raw],
        axis: 0,
      }}
    ]
  }
} + trainconfig
