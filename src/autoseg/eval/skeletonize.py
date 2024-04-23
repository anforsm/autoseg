import os
import sys
import numpy as np
import funlib.persistence
import kimimaro
import networkx as nx

from autoseg.datasets.load_dataset import get_dataset_path


def skeletonize(labels_file, labels_ds, teasar_params=None):
    if teasar_params is None:
        teasar_params = {
            "scale": 1.5,
            "const": 300,  # physical units
            "pdrf_scale": 100000,
            "pdrf_exponent": 4,
            "soma_acceptance_threshold": 3500,  # physical units
            "soma_detection_threshold": 750,  # physical units
            "soma_invalidation_const": 300,  # physical units
            "soma_invalidation_scale": 2,
            "max_paths": 300,  # default None
        }

    labels = funlib.persistence.open_ds(labels_file, labels_ds)
    roi = labels.roi
    labels_arr = labels.to_ndarray(roi=roi)
    vs = tuple(labels.voxel_size)

    print("starting to skeletonize")
    skels = kimimaro.skeletonize(
        labels_arr,
        teasar_params,
        # object_ids=[ ... ], # process only the specified labels
        # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
        # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
        dust_threshold=25,  # skip connected components with fewer than this many voxels
        anisotropy=vs,  # default True
        fix_branching=True,  # default True
        fix_borders=True,  # default True
        fill_holes=False,  # default False
        fix_avocados=False,  # default False
        progress=True,  # default False, show progress bar
        parallel=10,  # <= 0 all cpu, 1 single process, 2+ multiprocess
        parallel_chunk_size=10,  # how many skeletons to process before updating progress bar
    )

    uniques = np.unique(labels_arr)
    uniques = uniques[uniques > 0]
    skel_ids = np.array(list(skels.keys()))
    check = np.isin(uniques, skel_ids)

    if False not in check:
        return "good", skels, roi
    else:
        print("missing skels!")
        print("s=", teasar_params["scale"])
        print("c=", teasar_params["const"])
        missing_ids = uniques[~check]
        print(f"missing ids: {missing_ids}")
        # print(f"missing id

        return f"bad_{len(missing_ids)}", skels, roi


def convert_to_nx(skels, roi):
    G = nx.Graph()
    node_offset = 0

    offset = roi.offset

    for skel in skels:
        skeleton = skels[skel]

        # Add nodes
        for vertex in skeleton.vertices:
            G.add_node(
                node_offset,
                id=skeleton.id,
                position_z=vertex[0] + offset[0],
                position_y=vertex[1] + offset[1],
                position_x=vertex[2] + offset[2],
            )

            node_offset += 1

        # Add edges
        for edge in skeleton.edges:
            adjusted_u = edge[0] + node_offset - len(skeleton.vertices)
            adjusted_v = edge[1] + node_offset - len(skeleton.vertices)
            G.add_edge(adjusted_u, adjusted_v, u=adjusted_u, v=adjusted_v)

    return G


if __name__ == "__main__":
    params = {
        "scale": 1.5,
        "const": 300,  # physical units
        "pdrf_scale": 100000,
        "pdrf_exponent": 4,
        "soma_acceptance_threshold": 3500,  # physical units
        "soma_detection_threshold": 750,  # physical units
        "soma_invalidation_const": 300,  # physical units
        "soma_invalidation_scale": 2,
        "max_paths": 300,  # default None
    }

    path = get_dataset_path("SynapseWeb/kh2015/oblique").as_posix().replace(".zip", "")
    skels = skeletonize(path, "labels_f_r_eroded", params)
    # skels = skeletonize(path, "labels_filtered_relabeled", params)
    # skels = skeletonize(path, "labels/s0", params)

    out_f = f"./skel_filtered.graphml"
    os.makedirs(os.path.dirname(out_f), exist_ok=True)

    G = convert_to_nx(skels[1], skels[2])
    print(f"writing..{out_f}")
    nx.write_graphml(G, out_f)
