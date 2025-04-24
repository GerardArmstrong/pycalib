from hscimproc import FrameGenerator
import cv2 as cv
import numpy as np
from sys import stdout
from scipy.spatial.transform import Rotation
import json
from types import SimpleNamespace


class TusqCharucoBoard(cv.aruco.CharucoBoard):

    def __init__(self):

        _dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

        pattern_size = 0.0195
        super().__init__((10, 10), pattern_size, pattern_size*0.6, _dict)


class TusqCharucoDetector(cv.aruco.CharucoDetector):

    def __init__(self):
        ch_params = cv.aruco.CharucoParameters()
        d_params = cv.aruco.DetectorParameters()
        r_params = cv.aruco.RefineParameters()

        d_params.useAruco3Detection = True
        d_params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX

        self.board = TusqCharucoBoard()

        super().__init__(self.board, charucoParams=ch_params,
                         detectorParams=d_params, refineParams=r_params)


class Camera:

    class PropertiesContainer(object):
        def __repr__(self):
            return '\n'.join({f'{k}: {v}' for k, v in self.__dict__.items()})

    def __init__(self, frame_generator, name=None, cameraMatrix=None, px_size=None):

        self.frames = frame_generator
        self.aruco_frames = {}
        self.intrinsics = self.PropertiesContainer()
        self.extrinsics = self.PropertiesContainer()
        self.is_calibrated_intrinsic = False
        self.is_calibrated_extrinsic = False
        self.intrinsics.cameraMatrix = cameraMatrix
        self.name = name if type(name) is str else "Unknown Camera"
        self.px_size = px_size

    def save(self, path):
        data = {'intrinsics': {}, 'extrinsics': {}}
        data['extrinsics']['rvec'] = self.extrinsics.rvec.flatten().tolist()
        data['extrinsics']['T'] = self.extrinsics.T.tolist()
        data['extrinsics']['flags'] = self.extrinsics.flags
        data['intrinsics']['flags'] = self.intrinsics.flags
        data['intrinsics']['cameraMatrix'] = self.intrinsics.cameraMatrix.tolist()
        data['intrinsics']['distCoeffs'] = self.intrinsics.distCoeffs.tolist()
        data['is_calibrated_intrinsic'] = self.is_calibrated_intrinsic
        data['is_calibrated_extrinsic'] = self.is_calibrated_extrinsic
        data['name'] = self.name

        jsondata = json.dumps(data)
        with open(path, 'w') as f:
            json.dump(jsondata, f)

    def detect_charuco_corners(self, hflip=False, show_img=False, include=None, exclude=None, ignored_ch_ids=[]):

        detector = TusqCharucoDetector()
        self.detector = detector

        self.aruco_frames = {}

        for i, frame in enumerate(self.frames):

            # if exclude is not None and i in exclude: continue
            # if include is not None and not i in include: continue

            stdout.write(
                f"Running AruCo detection on {self.name} - frame {i}\r")

            frame = frame.copy()
            framergb = cv.cvtColor(frame, cv.COLOR_GRAY2BGR).copy()
            # detector = TusqCharucoDetector()

            ch_cnrs, ch_ids, marker_cnrs, marker_ids = detector.detectBoard(
                frame)

            if ch_cnrs is None:
                ch_cnr_points_world = None
            else:

                not_ignored = [_id not in ignored_ch_ids for _id in ch_ids]
                ch_ids = ch_ids[not_ignored]
                ch_cnrs = ch_cnrs[not_ignored]

                ch_cnr_points_world = detector.getBoard().getChessboardCorners()[
                    ch_ids]

                if show_img:
                    cv.aruco.drawDetectedCornersCharuco(
                        framergb, ch_cnrs, ch_ids, cornerColor=(0, 0, 255))
                    cv.putText(
                        framergb, f"Frame {i}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0))
                    cv.imshow(f'Detected corners for {self.name}', framergb)
                    cv.waitKey()

            self.add_detected_charuco_frame(
                i, ch_cnrs, ch_cnr_points_world, ch_ids)

        cv.destroyAllWindows()
        stdout.write('\n')

        if len(self.stack_charuco_frames()[0]) < 10:
            raise ValueError("Less than 10 images detected any features")

    def calibrate_intrinsic(self, include=None, exclude=None, err_threshold=3, flags=0, *args, **kwargs):

        assert len(self.aruco_frames.values()) > 10

        all_pts, all_pts_world, _ = self.stack_charuco_frames(
            include=include, exclude=exclude)

        distCoeffs = [0, 0, 0, 0, 0]

        retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv.calibrateCameraExtended(
            all_pts_world,
            all_pts,
            self.frames.im_shape,
            self.intrinsics.cameraMatrix,
            None,
            flags=flags
        )

        self.intrinsics.cameraMatrix = cameraMatrix
        self.intrinsics.distCoeffs = distCoeffs
        self.intrinsics.rvecs = rvecs
        self.intrinsics.retval = retval
        self.intrinsics.tvecs = tvecs
        self.intrinsics.flags = flags
        self.intrinsics.stdDeviationsIntrinsics = stdDeviationsIntrinsics
        self.intrinsics.stdDeviationsExtrinsics = stdDeviationsExtrinsics
        self.intrinsics.perViewErrors = perViewErrors

        bad_frames = self.map_to_frame_number(perViewErrors)
        bad_frames = {k: v for k, v in bad_frames.items() if v > err_threshold}

        if len(bad_frames.values()) > 0:
            print()
            print(
                "Consider adding the following frames to the exclude list (error shown alongside): ")
            print()
            for frame_num, err in bad_frames.items():
                print(f"{frame_num}: {err[0]:.2f}")
            print()

        if self.intrinsics.retval < err_threshold:
            self.is_calibrated_intrinsic = True

    def map_to_frame_number(self, l):

        ret = {}
        counter = 0
        for item in l:
            if self.aruco_frames[counter][0] is not None:
                ret[counter] = item
            counter += 1
        return ret

    def calibrate_extrinsic_with_solvepnp(self, camera2):

        camera1 = self

        assert camera1.is_calibrated_intrinsic
        assert camera2.is_calibrated_intrinsic

        camera1_cnrs, camera2_cnrs, world_cnrs = camera1.match_detected_points(
            camera2)
        # only keep the biggest few
        sindx = [len(x) for x in camera1_cnrs]
        sindx = np.argsort(sindx)[-1:-2:-1]
        camera1_cnrs = [x for i, x in enumerate(camera1_cnrs) if i in sindx]
        camera2_cnrs = [x for i, x in enumerate(camera2_cnrs) if i in sindx]
        world_cnrs = [x for i, x in enumerate(world_cnrs) if i in sindx]

        ret1, rvecs1, tvecs1, reproj1 = cv.solvePnPGeneric(
            objectPoints=world_cnrs[0],
            imagePoints=camera1_cnrs[0],
            cameraMatrix=camera1.intrinsics.cameraMatrix,
            distCoeffs=camera1.intrinsics.distCoeffs,
            flags=cv.SOLVEPNP_IPPE
        )

        ret1, rvecs2, tvecs2, reproj2 = cv.solvePnPGeneric(
            objectPoints=world_cnrs[0],
            imagePoints=camera2_cnrs[0],
            cameraMatrix=camera2.intrinsics.cameraMatrix,
            distCoeffs=camera2.intrinsics.distCoeffs,
            flags=cv.SOLVEPNP_IPPE
        )

        R1 = Rotation.from_rotvec(rvecs1[0].flatten())
        R2 = Rotation.from_rotvec(rvecs2[1].flatten())

        M1 = np.zeros((4, 4))
        M2 = np.zeros((4, 4))

        M1[:3, :3] = R1.as_matrix()
        M1[:3, 3] = tvecs1[0].flatten()
        M1[3, 3] = 1

        M2[:3, :3] = R2.as_matrix()
        M2[:3, 3] = tvecs2[1].flatten()
        M2[3, 3] = 1

        offset = M2 @ np.linalg.inv(M1)

        R = Rotation.from_matrix(offset[:3, :3]).as_matrix()
        t = offset[:3, 3]

        camera1.is_calibrated_extrinsic = True
        camera2.is_calibrated_extrinsic = True

        camera1.extrinsics.R = np.eye(3)
        camera1.extrinsics.T = np.array([0., 0., 0.])
        camera1.extrinsics.rvec = np.zeros(3)
        camera2.extrinsics.R = R
        camera2.extrinsics.T = t
        camera2.extrinsics.rvec = cv.Rodrigues(R)[0].flatten()

        del camera1.detector
        del camera2.detector
        del camera1.frames
        del camera2.frames

        return R, t

    def calibrate_extrinsic(self, camera2, n_best_images=1, R=None, T=None, flags=0):

        camera1 = self

        assert camera1.is_calibrated_intrinsic
        assert camera2.is_calibrated_intrinsic

        camera1_cnrs, camera2_cnrs, world_cnrs = camera1.match_detected_points(
            camera2)

        # only keep the biggest few
        sindx = [len(x) for x in camera1_cnrs]
        sindx = np.argsort(sindx)[-1:-(n_best_images+1):-1]
        camera1_cnrs = [x for i, x in enumerate(camera1_cnrs) if i in sindx]
        camera2_cnrs = [x for i, x in enumerate(camera2_cnrs) if i in sindx]
        world_cnrs = [x for i, x in enumerate(world_cnrs) if i in sindx]

        retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F, rvecs, tvecs, perViewErrors = cv.stereoCalibrateExtended(
            objectPoints=world_cnrs,
            imagePoints1=camera1_cnrs,
            imagePoints2=camera2_cnrs,
            cameraMatrix1=camera1.intrinsics.cameraMatrix,
            distCoeffs1=camera1.intrinsics.distCoeffs,
            cameraMatrix2=camera2.intrinsics.cameraMatrix,
            distCoeffs2=camera2.intrinsics.distCoeffs,
            imageSize=self.frames.im_shape,
            R=R,
            T=T,
            flags=flags,
        )

        # Camera one is considered to lie at origin
        self.extrinsics.retval = retval
        self.extrinsics.cameraMatrix1 = cameraMatrix1
        self.extrinsics.distCoeffs = distCoeffs1
        self.extrinsics.R = np.eye(3)
        self.extrinsics.T = np.zeros(3)
        self.extrinsics.rvec = cv.Rodrigues(np.eye(3))[0]
        self.extrinsics.E = E.T
        self.extrinsics.F = F.T
        self.extrinsics.rvecs = [-r for r in rvecs]
        self.extrinsics.tvecs = [-t for t in tvecs]
        self.extrinsics.perViewErrors = perViewErrors

        # Camera 2 will save all transformations as relative to camera 1
        camera2.extrinsics.retval = retval
        camera2.extrinsics.cameraMatrix1 = cameraMatrix2
        camera2.extrinsics.distCoeffs = distCoeffs2
        # R is the transform from Stereo camera 1 (TOP) to Stereo camera 2 (EAST). It is a change from world to east cam local coordinates.
        # The docs also say that it is Camera 1 position WRT Camera 2 - hence again, sounds like a change from world to local
        # Note that as a homogeneous transform the translation and rotation should be same CS. So if the rotation takes EAST axes to TOP axes that means the translation must be using EAST COORDINATES too, that is, T is expressed in camera 2's frame.
        camera2.extrinsics.R = R
        camera2.extrinsics.rvec = cv.Rodrigues(R)[0]
        camera2.extrinsics.T = T
        camera2.extrinsics.E = E
        camera2.extrinsics.F = F
        camera2.extrinsics.rvecs = rvecs
        camera2.extrinsics.tvecs = tvecs
        camera2.extrinsics.perViewErrors = perViewErrors

        camera1.is_calibrated_extrinsic = True
        camera2.is_calibrated_extrinsic = True

        # Delete any custom classes, otherwise pickle will fail on Camera object
        del camera1.detector
        del camera2.detector
        del self.frames
        del camera2.frames

        print(f"\nDone extrinsic calibration...\n\nRMS error: {retval}\n")

    def __repr__(self):
        return self.intrinsics.__repr__() + '\n\n' + self.extrinsics.__repr__()

    def add_detected_charuco_frame(self, frame_number, im_pts, world_pts, cnr_ids, cnr_thresh=20):
        # Used to save data from an aruco detection on an image

        if im_pts is not None and len(im_pts) < cnr_thresh:
            im_pts = None
            world_pts = None
            cnr_ids = None

        self.aruco_frames[frame_number] = [im_pts, world_pts, cnr_ids]

    def stack_charuco_frames(self, include=None, exclude=None):
        # Used to turn detections into a form suitable for drawing, or matching points by excluding null detections
        stripped_cnrs = []
        stripped_world_cnrs = []
        stripped_ids = []
        for frame_number, (im_cnrs, world_cnrs, _ids) in self.aruco_frames.items():

            if include is not None and not frame_number in include:
                continue
            if exclude is not None and frame_number in exclude:
                continue

            if im_cnrs is not None and _ids is not None and len(im_cnrs) >= 4:
                stripped_cnrs.append(im_cnrs)
                stripped_world_cnrs.append(world_cnrs)
                stripped_ids.append(_ids)

        return (stripped_cnrs, stripped_world_cnrs, stripped_ids)

    def match_detected_points(self, camera2):
        """
        Extrinsic calibration detected point matching
        """

        cam1_all_cnrs = []
        cam2_all_cnrs = []
        all_world_cnrs = []

        for frame_num, (im_cnrs, world_cnrs, _ids) in self.aruco_frames.items():

            if im_cnrs is None or _ids is None:
                continue

            if camera2.has_data_for_frame_number(frame_num):

                this_frame_cam1_cnrs = []
                this_frame_cam2_cnrs = []
                this_frame_cam1_ids = []
                this_frame_cam2_ids = []
                this_frame_world_cnrs = []

                for i, camera1_id in enumerate(_ids):

                    if camera2.has_id_for_frame_number(frame_num, camera1_id):
                        camera2_cnrs, camera2_world_cnrs, cam2_id = camera2.get_cnrs_by_id_and_frame(
                            frame_num, camera1_id)
                        camera1_cnrs = im_cnrs[i]

                        this_frame_cam1_cnrs.append(camera1_cnrs)
                        this_frame_cam1_ids.append(camera1_id)
                        this_frame_cam2_ids.append(cam2_id)
                        this_frame_cam2_cnrs.append(camera2_cnrs)
                        this_frame_world_cnrs.append(world_cnrs[i])

                if len(this_frame_cam1_cnrs) >= 4:
                    this_frame_cam1_cnrs = np.vstack(
                        this_frame_cam1_cnrs).reshape(-1, 1, 2)
                    this_frame_cam2_cnrs = np.vstack(
                        this_frame_cam2_cnrs).reshape(-1, 1, 2)
                    this_frame_world_cnrs = np.vstack(
                        this_frame_world_cnrs).reshape(-1, 1, 3)
                    this_frame_cam1_ids = np.vstack(this_frame_cam1_ids)
                    this_frame_cam2_ids = np.vstack(this_frame_cam2_ids)
                    cam1_all_cnrs.append(this_frame_cam1_cnrs)
                    cam2_all_cnrs.append(this_frame_cam2_cnrs)
                    all_world_cnrs.append(this_frame_world_cnrs)

                    # if len(this_frame_cam2_cnrs) > 4:
                    #     self.show_frame(frame_num=frame_num,cnrs=this_frame_cam1_cnrs,ids=this_frame_cam1_ids,winname='cam1')
                    #     camera2.show_frame(frame_num=frame_num,cnrs=this_frame_cam2_cnrs,ids=this_frame_cam2_ids,winname='cam2')
                    #     cv.waitKey()

        return (cam1_all_cnrs, cam2_all_cnrs, all_world_cnrs)

    def has_data_for_frame_number(self, frame_number):
        return True if self.aruco_frames[frame_number][0] is not None else False

    def has_id_for_frame_number(self, frame_number, _id):
        return True if _id in self.aruco_frames[frame_number][2] else False

    def get_cnrs_by_id_and_frame(self, frame_number, _id):
        for i, __id in enumerate(self.aruco_frames[frame_number][2]):
            if __id == _id:
                cnrs = self.aruco_frames[frame_number][0][i]
                world_cnrs = self.aruco_frames[frame_number][1][i]
                return cnrs, world_cnrs, __id

    def show_frame(self, frame_num, delay=0, cnrs=None, ids=None, winname='Frame'):
        frame = self.frames.get_frame(frame_num)
        frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

        if cnrs is not None:
            cv.aruco.drawDetectedCornersCharuco(frame, cnrs, ids, (0, 0, 255))

        cv.putText(frame, f"Frame: {frame_num}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        cv.imshow(winname, frame)

    def calc_sensor_size(self, x_res, y_res, px_size: np.array):
        """
        x_res: x-axis pixel resolution of sensor
        y_res: y-axis pixel resolution of sensor
        px_size: Size (mm) of single pixel

        Returns:
            Size of sensor in same units as px_size
        """

        assert self.is_calibrated_intrinsic

        focal_len_x = x_res * px_size
        focal_len_y = y_res * px_size

        return np.array([focal_len_x, focal_len_y])

    @property
    def R_as_quat(self):

        assert self.is_calibrated_extrinsic

        return Rotation.from_matrix(self.extrinsics.R).as_quat(scalar_first=True)
