from typing import Mapping, Optional, Sequence, Text, Tuple, Union
import numpy as np

from utils import camera_utils

# This is ugly, but it works.
import sys
sys.path.insert(0,'datasets/pycolmap')
sys.path.insert(0,'datasets/pycolmap/pycolmap')
import pycolmap

class NeRFSceneManager(pycolmap.SceneManager):
    """COLMAP pose loader.

    Minor NeRF-specific extension to the third_party Python COLMAP loader:
    google3/third_party/py/pycolmap/scene_manager.py
    """

    def process(
        self
    ) -> Tuple[Sequence[Text], 
               np.ndarray, 
               np.ndarray, 
               Sequence[Optional[Mapping[Text, float]]], 
               Sequence[camera_utils.ProjectionType],
               np.ndarray]:
        """Applies NeRF-specific postprocessing to the loaded pose data.

        Returns:
            a tuple [image_names, poses, pixtocam, distortion_params].
            image_names:  contains the only the basename of the images.
            poses: [N, 4, 4] array containing the camera to world matrices.
            pixtocam: [N, 3, 3] array containing the camera to pixel space matrices.
            distortion_params: mapping of distortion param name to distortion
                parameters. Cameras share intrinsics. Valid keys are k1, k2, p1 and p2.
        """

        self.load()

        # Extract intrinsics and distortion parameters and camtype
        camdata = self.cameras
        pixtocams = []
        distortion_params = []
        camtypes = []
        for k in camdata:
            cam = camdata[k]
            # Extract focal lengths and principal point parameters.
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            pixtocams.append(np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy)))

            type_ = cam.camera_type

            if type_ == 0 or type_ == 'SIMPLE_PINHOLE':
                params = None
                camtype = camera_utils.ProjectionType.PERSPECTIVE

            elif type_ == 1 or type_ == 'PINHOLE':
                params = None
                camtype = camera_utils.ProjectionType.PERSPECTIVE

            if type_ == 2 or type_ == 'SIMPLE_RADIAL':
                params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
                params['k1'] = cam.k1
                camtype = camera_utils.ProjectionType.PERSPECTIVE

            elif type_ == 3 or type_ == 'RADIAL':
                params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
                params['k1'] = cam.k1
                params['k2'] = cam.k2
                camtype = camera_utils.ProjectionType.PERSPECTIVE

            elif type_ == 4 or type_ == 'OPENCV':
                params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
                params['k1'] = cam.k1
                params['k2'] = cam.k2
                params['p1'] = cam.p1
                params['p2'] = cam.p2
                camtype = camera_utils.ProjectionType.PERSPECTIVE

            elif type_ == 5 or type_ == 'OPENCV_FISHEYE':
                params = {k: 0. for k in ['k1', 'k2', 'k3', 'k4']}
                params['k1'] = cam.k1
                params['k2'] = cam.k2
                params['k3'] = cam.k3
                params['k4'] = cam.k4
                camtype = camera_utils.ProjectionType.FISHEYE
            
            distortion_params.append(params)
            camtypes.append(camtype)

        # Extract extrinsic matrices in world-to-camera format.
        imdata = self.images
        w2c_mats = []
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)
        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        poses = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        names = [imdata[k].name for k in imdata]

        # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
        poses = poses @ np.diag([1, -1, -1, 1])

        if len(pixtocams) != len(names):
            if len(pixtocams) == 1:
                pixtocams = [pixtocams[0] for _ in range(len(names))]
                distortion_params = [distortion_params[0] for _ in range(len(names))]
                camtypes = [camtypes[0] for _ in range(len(names))]
            else:
                raise ValueError()
        pixtocams = np.stack(pixtocams, axis=0)

        # Extract pts3d
        pts3d = self.points3D
        
        return names, poses, pixtocams, distortion_params, camtypes, pts3d