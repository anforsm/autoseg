{
  predict: {
    multi_gpu: false,
    num_workers: 6,
    shape_increase: [0, 80, 80],
    source: [
      {
        path: "/home/anton/.cache/autoseg/datasets/SynapseWeb/kh2015/data/spine.zarr",
        dataset: "raw/s0"
      }
    ],

    output: [
      {
        path: "predict_3gpu.zarr",
        dataset: "preds/affs",
        num_channels: 3,
        #stacked: true,
      },
    ],
  }
}
