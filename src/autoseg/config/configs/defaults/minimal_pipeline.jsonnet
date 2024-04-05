


{
  pipeline: {
    local pad = import "autoseg/defaults/pad",
    local zarrsource = import "autoseg/defaults/zarrsource",

    _order: ["source", "normalize", "target", "post"],
    _outputs: ["RAW", "LABELS", "GT_AFFS", "AFFS_WEIGHTS", "GT_AFFS_MASK", "LABELS_MASK"],

    source: [
      [zarrsource.zarr_source("SynapseWeb/kh2015/oblique")] + pad,
    ],

    normalize: [
      {normalize: {
        array: "RAW"
      }},
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
