import argparse

import torch
import numpy as np
import random

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from geotransformer.utils.registration import compute_registration_error

from config import make_cfg
from model import create_model
import onnxruntime
import onnx


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", required=True, help="src point cloud numpy file")
    parser.add_argument("--ref_file", required=True, help="src point cloud numpy file")
    parser.add_argument("--gt_file", required=True, help="ground-truth transformation file")
    parser.add_argument("--weights", required=True, help="model weights file")
    return parser


def random_rotation_matrix():
    theta = 2 * np.pi * np.random.random()  # random rotation angle
    vector = np.random.randn(3)  # random vector
    vector /= np.linalg.norm(vector)  # convert to a unit vector

    # Compute a skew-symmetric matrix for the vector
    skew_symmetric = np.array([[0, -vector[2], vector[1]],
                               [vector[2], 0, -vector[0]],
                               [-vector[1], vector[0], 0]])

    # Compute the rotation matrix using the Rodriguez formula
    R = np.eye(3) + np.sin(theta) * skew_symmetric + (1 - np.cos(theta)) * np.dot(skew_symmetric, skew_symmetric)
    return R


def load_data(args):
    src_points = np.load(args.src_file)
    R = random_rotation_matrix()
    src_points = src_points @ R
    src_points += .5 * (.5 - np.random.rand(1, 3))

    ref_points = np.load(args.ref_file)
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
    }

    if args.gt_file is not None:
        transform = np.load(args.gt_file)
        data_dict["transform"] = transform.astype(np.float32)

    # feats = data_dict['features'].detach()
    # ref_length_c = data_dict['lengths'][-1][0].item()
    # ref_length_f = data_dict['lengths'][1][0].item()
    # ref_length = data_dict['lengths'][0][0].item()
    # points_c = data_dict['points'][-1].detach()
    # points_f = data_dict['points'][1].detach()
    # points = data_dict['points'][0].detach()
    # points_list = data_dict['points']
    # neighbors_list = data_dict['neighbors']
    # subsampling_list = data_dict['subsampling']
    # upsampling_list = data_dict['upsampling']

    return data_dict


def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()

    # prepare data
    data_dict = load_data(args)
    neighbor_limits = [38, 36, 36, 38]  # default setting in 3DMatch
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
    )

    # prepare model
    model = create_model(cfg).cpu()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict["model"])

    points_0 = data_dict['points'][0]
    points_1 = data_dict['points'][1]
    points_2 = data_dict['points'][2]
    points_3 = data_dict['points'][3]
    lengths_0 = data_dict['lengths'][0]
    lengths_1 = data_dict['lengths'][1]
    lengths_2 = data_dict['lengths'][2]
    lengths_3 = data_dict['lengths'][3]
    neighbors_0 = data_dict['neighbors'][0]
    neighbors_1 = data_dict['neighbors'][1]
    neighbors_2 = data_dict['neighbors'][2]
    neighbors_3 = data_dict['neighbors'][3]
    subsampling_0 = data_dict['subsampling'][0]
    subsampling_1 = data_dict['subsampling'][1]
    subsampling_2 = data_dict['subsampling'][2]
    upsampling_0 = data_dict['upsampling'][0]
    upsampling_1 = data_dict['upsampling'][1]
    upsampling_2 = data_dict['upsampling'][2]

    torch.onnx.export(model, args=(
        points_0, points_1, points_2, points_3, lengths_0, lengths_1, lengths_2, lengths_3, neighbors_0, neighbors_1,
        neighbors_2, neighbors_3, subsampling_0, subsampling_1,
        subsampling_2, upsampling_0, upsampling_1, upsampling_2),
                      f='network.onnx',
                      opset_version=17,
                      input_names=['points_0', 'points_1', 'points_2', 'points_3', 'lengths_0', 'lengths_1',
                                   'lengths_2', 'lengths_3', 'neighbors_0', 'neighbors_1',
                                   'neighbors_2', 'neighbors_3', 'subsampling_0', 'subsampling_1',
                                   'subsampling_2', 'upsampling_0', 'upsampling_1', 'upsampling_2'],
                      dynamic_axes={'points_0': {0: 'num_points_0'},  # variable length axes
                                    'points_1': {0: 'num_points_1'},
                                    'points_2': {0: 'num_points_2'},
                                    'points_3': {0: 'num_points_3'},
                                    'neighbors_0': {0: 'num_points_0'},
                                    'neighbors_1': {0: 'num_points_1'},
                                    'neighbors_2': {0: 'num_points_2'},
                                    'neighbors_3': {0: 'num_points_3'},
                                    'subsampling_0': {0: 'num_points_0'},
                                    'subsampling_1': {0: 'num_points_1'},
                                    'subsampling_2': {0: 'num_points_2'},
                                    'upsampling_0': {0: 'num_points_0'},
                                    'upsampling_1': {0: 'num_points_1'},
                                    'upsampling_2': {0: 'num_points_2'},
                                    },
                      output_names=['ref_node_corr_knn_points', 'src_node_corr_knn_points', 'ref_node_corr_knn_masks',
                                    'src_node_corr_knn_masks', 'matching_scores', 'node_corr_scores']
                      )


if __name__ == "__main__":
    main()
