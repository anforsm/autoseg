{
  pipeline: {
    local raw = "RAW",
    local labels = "LABELS",
    local labels_mask = "LABELS_MASK",
    local affinities = "GT_AFFS",
    local affinities_mask = "GT_AFFS_MASK",
    local affinities_weights = "AFFS_WEIGHTS",
    local non_interpolatable_array_spec = {array_spec: {
      interpolatable: false
    }},
    local interpolatable_array_spec = {array_spec: {
      interpolatable: true
    }},
    _order: ["source", "normalize", "augment", "target"],

    source: [
      {zarr_source: {
        store: "SynapseWeb/kh2015/oblique",
        datasets: {
          _raw: "raw/s0",
          _labels: "labels/s0",
          _labels_mask: "labels_mask/s0",
        },
        array_specs: {
          _raw: interpolatable_array_spec,
          _labels: non_interpolatable_array_spec,
          _labels_mask: non_interpolatable_array_spec,
        },
      }},
      {pad: {
        key: raw,
        size: null,
      }},
      {pad: {
        key: labels,
        size: {coordinate: {
          _positional: [450, 290, 290]
        }},
      }},
      {pad: {
        key: labels_mask,
        size: {coordinate: {
          _positional: [450, 290, 290]
        }},
      }},
      {random_location: {
        mask: labels_mask,
        min_masked: 0.1
      }}
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
  },
  training: {
    optimizer: {
      name: "adam",
      learning_rate: 1e-4,
    },
    batch_size: 10,
    loss: "affinities_sigmoid",
  }
}
