[
  {pad: {
    key: "RAW",
    size: null,
    mode: "reflect",
  }},
  {pad: {
    key: "LABELS",
    size: {coordinate: {
      _positional: [300, 280, 280]
      #_positional: [300, 400, 400]
    }},
  }},
  {pad: {
    key: "LABELS_MASK",
    size: {coordinate: {
      _positional: [300, 280, 280]
      #_positional: [300, 400, 400]
    }},
  }},
  {random_location: {
    mask: "LABELS_MASK",
    min_masked: 0.7,
  }}
]
