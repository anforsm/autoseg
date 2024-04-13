{
  predict: {
    multi_gpu: false,
    num_workers: 6,
    shape_increase: [0, 800, 800],
    use_latest: true,
    source: [
      {
        #path: "/home/anton/.cache/autoseg/datasets/SynapseWeb/kh2015/data/spine.zarr",
        path: "SynapseWeb/kh2015/oblique",
        dataset: "raw/s0",
      }
    ],

    output: [
      {
        path: "./oblique_prediction.zarr",
        dataset: "preds/affs",
        num_channels: 3,
        #stacked: true,
      },
    ],
  }
}
