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
      _raw: "raw/s0",
      _labels: "labels/s0",
      _labels_mask: "labels_mask/s0",
    },
    array_specs: {
      _raw: interpolatable_array_spec,
      _labels: non_interpolatable_array_spec,
      _labels_mask: non_interpolatable_array_spec,
    }
  }},
}
