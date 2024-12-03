import argparse

import rclcpp
import torch
import numpy as np

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from geotransformer.utils.registration import compute_registration_error

from config import make_cfg
from model import create_model
from scipy.spatial.transform import Rotation as R


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="model weights file")
    return parser


def make_parser_test():
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

    return data_dict


def run_model(ort_sess, model, ref_points, src_points):
    num_stages = 4
    neighbor_limits = [38, 36, 36, 38]
    init_voxel_size = 0.025
    init_radius = 0.0625
    data_dict = {}
    data_dict['ref_points'] = ref_points
    data_dict['src_points'] = src_points
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict["ref_feats"] = ref_feats.astype(np.float32)
    data_dict["src_feats"] = src_feats.astype(np.float32)

    data_dict = registration_collate_fn_stack_mode(
        [data_dict], num_stages, init_voxel_size, init_radius, neighbor_limits
    )

    run_onnx = False
    if run_onnx:
        points_0 = data_dict['points'][0].cpu().numpy()
        points_1 = data_dict['points'][1].cpu().numpy()
        points_2 = data_dict['points'][2].cpu().numpy()
        points_3 = data_dict['points'][3].cpu().numpy()
        lengths_0 = data_dict['lengths'][0].cpu().numpy()
        lengths_1 = data_dict['lengths'][1].cpu().numpy()
        lengths_2 = data_dict['lengths'][2].cpu().numpy()
        lengths_3 = data_dict['lengths'][3].cpu().numpy()
        neighbors_0 = data_dict['neighbors'][0].cpu().numpy()
        neighbors_1 = data_dict['neighbors'][1].cpu().numpy()
        neighbors_2 = data_dict['neighbors'][2].cpu().numpy()
        neighbors_3 = data_dict['neighbors'][3].cpu().numpy()
        subsampling_0 = data_dict['subsampling'][0].cpu().numpy()
        subsampling_1 = data_dict['subsampling'][1].cpu().numpy()
        subsampling_2 = data_dict['subsampling'][2].cpu().numpy()
        upsampling_0 = data_dict['upsampling'][0].cpu().numpy()
        upsampling_1 = data_dict['upsampling'][1].cpu().numpy()
        upsampling_2 = data_dict['upsampling'][2].cpu().numpy()

        ref_node_corr_knn_points, src_node_corr_knn_points, ref_node_corr_knn_masks, src_node_corr_knn_masks, matching_scores, node_corr_scores = ort_sess.run(
            None,
            {'points_0': points_0, 'points_1': points_1, 'points_2': points_2, 'points_3': points_3,
             'lengths_1': lengths_1, 'lengths_3': lengths_3, 'neighbors_0': neighbors_0,
             'neighbors_1': neighbors_1,
             'neighbors_2': neighbors_2, 'neighbors_3': neighbors_3, 'subsampling_0': subsampling_0,
             'subsampling_1': subsampling_1,
             'subsampling_2': subsampling_2, 'upsampling_1': upsampling_1,
             'upsampling_2': upsampling_2})

    else:
        points_0 = data_dict['points'][0].cuda()
        points_1 = data_dict['points'][1].cuda()
        points_2 = data_dict['points'][2].cuda()
        points_3 = data_dict['points'][3].cuda()
        lengths_0 = data_dict['lengths'][0].cuda()
        lengths_1 = data_dict['lengths'][1].cuda()
        lengths_2 = data_dict['lengths'][2].cuda()
        lengths_3 = data_dict['lengths'][3].cuda()
        neighbors_0 = data_dict['neighbors'][0].cuda()
        neighbors_1 = data_dict['neighbors'][1].cuda()
        neighbors_2 = data_dict['neighbors'][2].cuda()
        neighbors_3 = data_dict['neighbors'][3].cuda()
        subsampling_0 = data_dict['subsampling'][0].cuda()
        subsampling_1 = data_dict['subsampling'][1].cuda()
        subsampling_2 = data_dict['subsampling'][2].cuda()
        upsampling_0 = data_dict['upsampling'][0].cuda()
        upsampling_1 = data_dict['upsampling'][1].cuda()
        upsampling_2 = data_dict['upsampling'][2].cuda()

        ref_node_corr_knn_points, src_node_corr_knn_points, ref_node_corr_knn_masks, src_node_corr_knn_masks, matching_scores, node_corr_scores = model(
        points_0, points_1, points_2, points_3, lengths_0, lengths_1, lengths_2, lengths_3, neighbors_0, neighbors_1,
        neighbors_2, neighbors_3, subsampling_0, subsampling_1,
        subsampling_2, upsampling_0, upsampling_1, upsampling_2)

# prediction

    matching_scores = matching_scores[:, :-1, :-1]
    # ref_node_corr_knn_points = ref_node_corr_knn_points.cpu()
    # src_node_corr_knn_points = src_node_corr_knn_points.cpu()


    ref_corr_points, src_corr_points, corr_scores, estimated_transform = model.fine_matching(
        ref_node_corr_knn_points,
        src_node_corr_knn_points,
        ref_node_corr_knn_masks,
        src_node_corr_knn_masks,
        matching_scores,
        node_corr_scores,
    )

    # get results
    ref_points = points_0[:lengths_0[0]]
    src_points = points_0[lengths_0[0]:]
    estimated_transform = estimated_transform

    return estimated_transform, ref_points, src_points


import rclpy
from rclpy.node import Node
from tutorial_interfaces.srv import Registration
import onnxruntime as ort


class GeotransformerROS(Node):

    def __init__(self, model):
        super().__init__('geotransformer')
        self.srv = self.create_service(Registration, 'geotransformer_registration', self.do_registration)
        self.model = model
        # self.ort_sess = ort.InferenceSession('network.onnx', providers=['CUDAExecutionProvider'])
        self.ort_sess = ort.InferenceSession('network.onnx')

    def do_registration(self, request, response):
        # extract data from input clouds
        print("got request")
        source_cloud = request.source_cloud
        target_cloud = request.target_cloud

        if len(target_cloud.data) == 0:
            #  use identity
            print("response returned: bad data")
            return response

        def get_points(msg):
            data = np.frombuffer(msg.data, dtype=np.uint8)
            data = np.reshape(data, (-1, 16))
            points_data = data[:, :12].copy()
            points_float = np.frombuffer(points_data, dtype=np.float32)
            points_float = points_float[~np.isnan(points_float)]
            points_float = np.reshape(points_float, (-1, 3))
            inds = np.linspace(0, points_float.shape[0] - 1, 2000, dtype=np.int32).flatten()
            data_float = points_float[inds, :]
            return data_float

        try:
            source_cloud_points = get_points(source_cloud)
            target_cloud_points = get_points(target_cloud)
        except Exception as e:
            print("Exception: ", e)
            return response

        try:
            estimated_transform, ref_points, src_points = run_model(self.ort_sess, self.model, source_cloud_points,
                                                                        target_cloud_points)
            estimated_transform = estimated_transform.detach().cpu().numpy()
            if np.linalg.norm(estimated_transform[:3,3]) > 0.3:
                print("bad norm")
                return response
        except Exception as e:
            print(e)
            return response

        response.estimated_transform.translation.x = estimated_transform[0, 3].item()
        response.estimated_transform.translation.y = estimated_transform[1, 3].item()
        response.estimated_transform.translation.z = estimated_transform[2, 3].item()

        rot = estimated_transform[:3, :3]
        # Convert rotation matrix to quaternion
        rotation = R.from_matrix(rot)
        quaternion = rotation.as_quat()

        response.estimated_transform.rotation.w = 1.0  # quaternion[3].item()
        response.estimated_transform.rotation.x = 0.0  # quaternion[0].item()
        response.estimated_transform.rotation.y = 0.0  # quaternion[1].item()
        response.estimated_transform.rotation.z = 0.0  # quaternion[2].item()

        print("response returned: ", response.estimated_transform)

        return response


def run_ros():
    parser = make_parser_test()
    args = parser.parse_args()

    # prepare model
    cfg = make_cfg()
    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict["model"])

    rclpy.init()
    node = GeotransformerROS(model)
    rclpy.spin(node)


def main():
    testing = False
    if testing:
        parser = make_parser_test()
        args = parser.parse_args()

        # prepare data
        data_dict = load_data(args)
        # prepare model
        cfg = make_cfg()
        model = create_model(cfg).cuda()
        state_dict = torch.load(args.weights)
        model.load_state_dict(state_dict["model"])

        estimated_transform, ref_points, src_points = run_model(model, data_dict['ref_points'],
                                                                data_dict['src_points'])

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
    else:
        run_ros()


if __name__ == "__main__":
    main()
