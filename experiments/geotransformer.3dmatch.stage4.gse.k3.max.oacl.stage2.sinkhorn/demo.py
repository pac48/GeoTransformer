import argparse

import torch
torch.manual_seed(0)

import numpy as np
np.random.seed(0)

import random
# Set the random seed
random.seed(0)

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from geotransformer.utils.registration import compute_registration_error

from config import make_cfg
from model import create_model
import onnx


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", required=True, help="src point cloud numpy file")
    parser.add_argument("--ref_file", required=True, help="src point cloud numpy file")
    parser.add_argument("--gt_file", required=True, help="ground-truth transformation file")
    parser.add_argument("--weights", required=True, help="model weights file")
    return parser


def load_data(args):
    src_points = np.load(args.src_file)
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

    # prediction
    # data_dict = to_cuda(data_dict)
    while True:
        output_dict = model(data_dict)
        print(output_dict['estimated_transform'])
        break

    # onnx_model = onnx.load('network.onnx')
    # graph = onnx_model.graph
    # onnx.checker.check_model(onnx_model)
    # import onnxruntime
    # # ort_session = onnxruntime.InferenceSession("network.onnx", providers=["CUDAExecutionProvider"])
    # # ort_session = onnxruntime.InferenceSession("network.onnx", providers=["CPUExecutionProvider"])
    # ort_session = onnxruntime.InferenceSession("network.onnx", providers=[
    #     # ('CUDAExecutionProvider', {
    #     #     'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
    #     # }),
    #     'CPUExecutionProvider',
    # ])
    # # missing lengths, upsample[0], and batch_size
    #
    # inputs = {'features': data_dict['features'].cpu().numpy(), 'points_0': data_dict['points'][0].cpu().numpy(),
    #           'points_1': data_dict['points'][1].cpu().numpy(),
    #           'points_2': data_dict['points'][2].cpu().numpy(), 'points_3': data_dict['points'][3].cpu().numpy(),
    #           # 'neighbors_0': data_dict['neighbors'][0], 'neighbors_1': data_dict['neighbors'][1],
    #           # 'neighbors_2': data_dict['neighbors'][2], 'neighbors_3': data_dict['neighbors'][3],
    #           'neighbors_0': data_dict['neighbors'][0].cpu().numpy(),
    #           'neighbors_1': data_dict['neighbors'][1].cpu().numpy(),
    #           'neighbors_2': data_dict['neighbors'][2].cpu().numpy(),
    #           'neighbors_3': data_dict['neighbors'][3].cpu().numpy(),
    #           'subsampling_0': data_dict['subsampling'][0].cpu().numpy(),
    #
    #           'neighbor_indices.7': data_dict['subsampling'][1].cpu().numpy(),
    #           'neighbor_indices.11': data_dict['subsampling'][2].cpu().numpy(),
    #           'onnx::Gather_17': data_dict['upsampling'][1].cpu().numpy(),
    #           'onnx::Gather_18': data_dict['upsampling'][2].cpu().numpy()}
    # while True:
    #     outputs = ort_session.run(None, inputs)
    #     print(outputs[20])
    #


    # exit(0)
    #
    # # with open('network.onnx', 'w') as f:
    # torch.onnx.export(model, {'data_dict': data_dict}, 'network.onnx', opset_version=17,
    #                   input_names=['features', 'points_0', 'points_1', 'points_2', 'points_3', 'subsampling_0',
    #                                'subsampling_1AAA',
    #                                'subsampling_2AAAA', 'upsampling_1', 'neighbors_0',
    #                                'neighbors_1', 'neighbors_2', 'neighbors_3', 'subsampling_0'])
    #
    #


    data_dict = release_cuda(data_dict)
    output_dict = release_cuda(output_dict)

    # # get results
    ref_points = output_dict["ref_points"]
    src_points = output_dict["src_points"]
    estimated_transform = output_dict["estimated_transform"]
    transform = data_dict["transform"]

    # visualization
    ref_pcd = make_open3d_point_cloud(ref_points)
    ref_pcd.estimate_normals()
    ref_pcd.paint_uniform_color(get_color("custom_yellow"))
    src_pcd = make_open3d_point_cloud(src_points)
    src_pcd.estimate_normals()
    src_pcd.paint_uniform_color(get_color("custom_blue"))
    draw_geometries(ref_pcd, src_pcd)
    src_pcd = src_pcd.transform(estimated_transform)
    draw_geometries(ref_pcd, src_pcd)

    # compute error
    rre, rte = compute_registration_error(transform, estimated_transform)
    print(f"RRE(deg): {rre:.3f}, RTE(m): {rte:.3f}")


if __name__ == "__main__":
    main()
