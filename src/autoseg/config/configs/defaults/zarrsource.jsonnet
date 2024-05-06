{zarr_source(store="SynapseWeb/kh2015/oblique", scale="s0") ::
  local non_interpolatable_array_spec = {array_spec: {
    interpolatable: false
  }};
  local interpolatable_array_spec = {array_spec: {
    interpolatable: true
  }};

  {zarr_source: {
    store: store,
    datasets: {
      _raw: "raw" + if scale != null then "/" + scale else "",
      _labels: "labels" + if scale != null then "/" + scale else "",
      _labels_mask: "labels_mask" + if scale != null then "/" + scale else "",
    },
    array_specs: {
      _raw: interpolatable_array_spec,
      _labels: non_interpolatable_array_spec,
      _labels_mask: non_interpolatable_array_spec,
    }
  }},
}
