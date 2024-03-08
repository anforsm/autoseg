{target: [
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
   {intensity_scale_shift: {
     array: "RAW",
     scale: 2,
     shift: -1
   }},
   {unsqueeze: {
     arrays: ["RAW"],
     axis: 0,
   }}
 ]}
