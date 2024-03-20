{zarr_source_2d(store="SynapseWeb/kh2015/oblique", index=0) ::
  local non_interpolatable_array_spec = {array_spec: {
    interpolatable: false
  }};
  local interpolatable_array_spec = {array_spec: {
    interpolatable: true
  }};

  {zarr_source: {
    store: store,
    datasets: {
      _raw: "raw/s0/" + index,
      _labels: "labels/s0/" + index,
      _labels_mask: "labels_mask/s0/" + index,
    },
    array_specs: {
      _raw: interpolatable_array_spec,
      _labels: non_interpolatable_array_spec,
      _labels_mask: non_interpolatable_array_spec,
    }
  }},
}
