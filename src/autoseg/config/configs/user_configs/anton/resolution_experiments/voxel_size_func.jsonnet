{get_pipeline_for_voxel_size(voxel_size=[50, 2, 2], name="UNet s0") ::
  local default = import "autoseg/user_configs/anton/resolution_experiments/default";
  local pad = import "autoseg/user_configs/anton/resolution_experiments/pad";
  local zarrsource = import "autoseg/user_configs/anton/resolution_experiments/zarrsource";

  default + {
    model+: {
      name: name,
    },
    pipeline+: {
      _voxel_size: voxel_size,

      source: [
        [zarrsource.zarr_source("SynapseWeb/kh2015/apical")]

        + [{resample: {
          source: arr + "_ORIGINAL",
          target: arr,
          target_voxel_size: {
            coordinate: {
              _positional: voxel_size
            }
          }
        }} for arr in ["RAW", "LABELS", "LABELS_MASK"]]

        + pad

        + [
          {random_location: {
            mask: "LABELS_MASK",
            min_masked: 0.7
          }}
        ]
      ],
    }
  }
}
