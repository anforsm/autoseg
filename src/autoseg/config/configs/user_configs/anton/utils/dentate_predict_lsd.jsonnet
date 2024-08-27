local datasets = ['BBCHZ', 'BLSNK', 'BRNLG', 'CLZBJ', 'CRQCR', 'CSKBZ', 'DRZNC', 'DTZVX', 'GFBZX', 'HNVRR', 'HVCBQ', 'JJPSC', 'KSGRS', 'MCZJJ', 'MFBCF', 'MPLTJ', 'NDKZB', 'QRFNB', 'RFHTC', 'RJZQR', 'RLCVK', 'SRQHN', 'TYLYL', 'YSJNL'];
{
  predict+: {
    "datasets": [
      {
        "name": dataset,
        "shape_increase": [
          -12,
          405,
          405
        ],
        "mask": {
          "path": "SynapseWeb/team_dentate/" + dataset,
          "dataset": "volumes/object_mask/s1",
          "use_in_postproc": false
        },
        "source": {
          "path": "SynapseWeb/team_dentate/" + dataset,
          "dataset": "volumes/image/s1"
        },
        "output": [
          {
            "path": dataset + "_prediction.zarr",
            "dataset": "preds/affs",
            "num_channels": 3
          },
          {
            "path": dataset + "_prediction.zarr",
            "dataset": "preds/lsds",
            "num_channels": 10
          }
        ]
      },
    for dataset in datasets]
  },
  evaluation+: {
    "datasets": [
      {
        "name": dataset,
        "ground_truth_skeletons": "SynapseWeb/team_dentate/" + dataset + ".graphml",
        "ground_truth_labels": {
          "path": "SynapseWeb/team_dentate/" + dataset,
          "dataset": "volumes/neuron_ids/s1"
        },
        "output": dataset + "_results.json"
      },
    for dataset in datasets]
  }
}
