import argparse
from pathlib import Path

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

from carla_data_utils import load_raw_data_infos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0, required=False)
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()

    root_path = Path(args.dir)

    data_infos = load_raw_data_infos(root_path, args.id)
    data_info = data_infos[0]

    print("scene_id", data_info.scene_id)

    points_list = []
    for lidar_info in data_info.lidars:
        lidar_info.pc_path

        pc = o3d.io.read_point_cloud(
            str(root_path / lidar_info.pc_path)
        )
        points = np.asarray(pc.points).astype(np.float32)

        # LiDARs to world
        pc.rotate(lidar_info.sensor_rot, center=[0., 0., 0.])
        pc.translate(lidar_info.sensor_trans)

        points = np.asarray(pc.points).astype(np.float32)
        points_list.append(points)

    # fuse four LiDARS
    points = np.concatenate(points_list, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    bboxes = []
    for vehicle in data_info.vehicles:
        bbox = vehicle.get_bbox()

        bottom_center = bbox[0:3]
        extent = bbox[3:6]
        yaw = bbox[6]


        center = bottom_center[:2] + [bottom_center[2] + extent[2] / 2.]

        # get rotation matrix
        r = R.from_euler("z", yaw).as_matrix()
        bbox = o3d.geometry.OrientedBoundingBox(
            center=center,
            R=r,
            extent=extent,
        )
        bbox.color = [1, 0, 0]
        bboxes.append(bbox)

    # transfer all the world elements to LiDAR0 coordinates
    world_to_lidar0_rot = np.linalg.inv(data_info.lidars[0].sensor_rot)
    world_to_lidar0_trans = -data_info.lidars[0].sensor_trans
    for geom in [pcd] + bboxes:
        geom.translate(world_to_lidar0_trans)
        geom.rotate(world_to_lidar0_rot, center=[0., 0., 0.])

    # left-hand to right-hand
    geoms = []
    for geom in [pcd] + bboxes:
        if isinstance(geom, o3d.geometry.OrientedBoundingBox):
            center = geom.center.copy()
            # y -> -y
            center[1] *= -1
            bbox = o3d.geometry.OrientedBoundingBox(
                center=center,
                R=geom.R,
                extent=geom.extent,
            )
            bbox.color = [1, 0, 0]
            geoms.append(bbox)
        elif isinstance(geom, o3d.geometry.PointCloud):
            points = np.asarray(geom.points).astype(np.float32)
            # y -> -y
            points[:, 1] *= -1
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            geoms.append(pcd)
        else:
            raise NotImplementedError(geom)

    o3d.visualization.draw(geoms)


if __name__ == "__main__":
    main()
