{zarr_source(store="SynapseWeb/kh2015/oblique") ::
  local non_interpolatable_array_spec = {array_spec: {
    interpolatable: false
  }};
  local interpolatable_array_spec = {array_spec: {
    interpolatable: true
  }};

  {zarr_source: {
    store: store,
    datasets: {
      _raw_original: "raw/s0",
      _labels_original: "labels/s0",
      _labels_mask_original: "labels_mask/s0",
    },
    array_specs: {
      _raw_original: interpolatable_array_spec,
      _labels_original: non_interpolatable_array_spec,
      _labels_mask_original: non_interpolatable_array_spec,
    }
  }},
}
