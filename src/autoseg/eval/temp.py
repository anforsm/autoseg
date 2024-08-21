import json

datasets = [
    "BLSNK",
    "BRNLG",
    "CLZBJ",
    "CRQCR",
    "CSKBZ",
    "DRZNC",
    "DTZVX",
    "GFBZX",
    "HNVRR",
    "HVCBQ",
    "JJPSC",
    "KSGRS",
    "MCZJJ",
    "MFBCF",
    "MPLTJ",
    "NDKZB",
    "QRFNB",
    "RFHTC",
    "RJZQR",
    "RLCVK",
    "SRQHN",
    "TYLYL",
    "YSJNL",
]

config = {
    "predict": {
        "datasets": [
            {
                "name": "BBCHZ",
                "shape_increase": [-12, 405, 405],
                "mask": {
                    "path": "SynapseWeb/team_dentate/BBCHZ",
                    "dataset": "volumes/object_mask/s1",
                    "use_in_postproc": False,
                },
                "source": {
                    "path": "SynapseWeb/team_dentate/BBCHZ",
                    "dataset": "volumes/image/s1",
                },
                "output": [
                    {
                        "path": "BBCHZ_prediction.zarr",
                        "dataset": "preds/affs",
                        "num_channels": 3,
                    }
                ],
            }
        ]
    },
    "evaluation": {
        "datasets": [
            {
                "name": "BBCHZ",
                "ground_truth_skeletons": "SynapseWeb/team_dentate/BBCHZ",
                "ground_truth_labels": {
                    "path": "SynapseWeb/team_dentate/BBCHZ",
                    "dataset": "volumes/neuron_ids/s1",
                },
                "output": "BBCHZ_results.json",
            }
        ]
    },
}

# Add new datasets to predict section
for dataset in datasets:
    new_dataset = {
        "name": dataset,
        "shape_increase": [-12, 405, 405],
        "mask": {
            "path": f"SynapseWeb/team_dentate/{dataset}",
            "dataset": "volumes/object_mask/s1",
            "use_in_postproc": False,
        },
        "source": {
            "path": f"SynapseWeb/team_dentate/{dataset}",
            "dataset": "volumes/image/s1",
        },
        "output": [
            {
                "path": f"{dataset}_prediction.zarr",
                "dataset": "preds/affs",
                "num_channels": 3,
            }
        ],
    }
    config["predict"]["datasets"].append(new_dataset)

# Add new datasets to evaluation section
for dataset in datasets:
    new_dataset = {
        "name": dataset,
        "ground_truth_skeletons": f"SynapseWeb/team_dentate/{dataset}",
        "ground_truth_labels": {
            "path": f"SynapseWeb/team_dentate/{dataset}",
            "dataset": "volumes/neuron_ids/s1",
        },
        "output": f"{dataset}_results.json",
    }
    config["evaluation"]["datasets"].append(new_dataset)

# Print the resulting configuration
print(json.dumps(config, indent=2))
