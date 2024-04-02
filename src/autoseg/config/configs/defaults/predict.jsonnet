{
  predict: {
    multi_gpu: true,
    num_workers: 3,
    source: [
      {
        path: "SynapseWeb/kh2015/spine",
        dataset: "raw/s0"
      }
    ],

    output: [
      {
        path: "multiout.zarr",
        dataset: "preds/affs",
        num_channels: 3,
        #stacked: true,
      },
    ],
  }
}
