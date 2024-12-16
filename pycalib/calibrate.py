"""
The plan:

- Calibrator is a superclass that is designed to handle calibration of images at TUSQ:
    - Perform an intrinsic calibration using our rig
    - This usually has a set number of images (360)
    - The first one should be discarded as we get an extra frame at start
    - Perform extrinsic calibrations too
    - Save the camera model, distortion coefficients etc
"""

from hscimproc import FrameGenerator
import cv2 as cv
import numpy as np

frames = None

frame_generator = FrameGenerator()

class TusqCharucoBoard(cv.aruco.CharucoBoard):

    def __init__(self):

        dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
        
        super().__init__((10,10),0.01915,0.01915*0.6,dict)

class TusqCharucoDetector(cv.aruco.CharucoDetector):
    

    def __init__(self):
        ch_params = cv.aruco.CharucoParameters()
        d_params = cv.aruco.DetectorParameters()
        r_params = cv.aruco.RefineParameters()

        d_params.useAruco3Detection = True
        d_params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX

        super().__init__(TusqCharucoBoard(),charucoParams=ch_params,detectorParams=d_params,refineParams=r_params)

class CameraCalibrator:

    def __init__(self,im_shape=(1024,1024)):
        # frame_generator = FrameGenerator()
        self.im_shape = im_shape

    def detect_charuco_corners(self,include=None,exclude=None,show_img=False):
        
        global frames

        detector = TusqCharucoDetector()

        all_pts = []
        all_pts_world = []
        all_ids = []
        used = []

        for i, frame in enumerate(frames):

            if exclude is not None and i in exclude: continue
            if include is not None and not i in include: continue

            print(f"Running AruCo detection - frame {i}")

            framergb = cv.cvtColor(frame,cv.COLOR_GRAY2BGR)

            ch_cnrs, ch_ids, marker_cnrs, marker_ids = detector.detectBoard(frame)

            if ch_cnrs is not None and len(ch_cnrs) >= 4:
                
                if show_img:
                    cv.aruco.drawDetectedCornersCharuco(framergb,ch_cnrs,ch_ids,(0,0,255))
                    cv.aruco.drawDetectedMarkers(framergb,marker_cnrs,marker_ids,(0,255,0))
                    cv.imshow('',framergb)
                    key = cv.waitKey(0)
                    if key == ord('q'): break

                ch_cnr_points_world = detector.getBoard().getChessboardCorners()[ch_ids]

                all_pts.append(ch_cnrs)
                all_pts_world.append(ch_cnr_points_world)
                all_ids.append(ch_ids)
                used.append(i)

        if len(all_pts) < 10:
            raise ValueError("Less than 10 images detected any features")
        
        return all_pts,all_pts_world,all_ids,used

    def _calibrate_intrinsic(self,frames,include=None,exclude=None,show_img=False,err_threshold=3,*args,**kwargs):
        
        all_pts,all_pts_world,all_ids,used = self.detect_charuco_corners(frames,include,exclude,show_img)

        retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv.calibrateCameraExtended(all_pts_world,all_pts,self.im_shape,None,None)

        self.retval = retval
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.stdDeviationsIntrinsics = stdDeviationsExtrinsics
        self.stdDeviationsExtrinsics = stdDeviationsExtrinsics
        self.perViewErrors = perViewErrors
        self.used = used

        bad_frames = [(used[j],err) for j, err in enumerate(perViewErrors) if err > err_threshold]

        if bad_frames:
            print()
            print("Consider adding the following frames to the exclude list (error shown alongside): ")
            print()
            for frame_num, err in bad_frames:
                print(f"{frame_num}: {err[0]:.2f}")
            print()

        return retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, used        

    def calibrate_from_mraw(self,mraw,include=None,exclude=None,hflip=False,*args,**kwargs):
        
        frames = frame_generator.raw_frame_generator_int8(mraw,hflip=hflip,*args,**kwargs)

        return self._calibrate_intrinsic(frames,include,exclude)

    def calibrate_from_image_files(self,glob_pattern,include=None,exclude=None,hflip=False,*args,**kwargs):

        frames = frame_generator.standard_format_frame_generator(glob_pattern,hflip=hflip,*args,**kwargs)
        print(id(frames))
        return self._calibrate_intrinsic(frames,include,exclude)

    def calibrate_extrinsic(self,frames_c1,frames_c2,cameraMatrix1,include=None,exclude=None):

        global frames
        cam_1_im_pts = []
        cam_2_im_pts = []

        frames_gen = FrameGenerator()
        frames = frames_gen.raw_frame_generator_int8(frames_c1,hflip=True)

        print("Running extrinsic calibration...")
        print("Detecting chessboard corners from Camera 1...")
        all_pts_c1,_,all_ids_c1,used_c1 = self.detect_charuco_corners(include=include,exclude=exclude)

        frames = frames_gen.raw_frame_generator_int8(frames_c2)

        print("Detecting chessboard corners from Camera 2...")
        all_pts_c2,_,all_ids_c2,used_c2 = self.detect_charuco_corners(include=include,exclude=exclude)          


        all_pts_c1 = {i: all_pts_c1[i] for i in all_ids_c1[0].flatten()}
        all_pts_c2 = {i: all_pts_c2[i] for i in all_ids_c2[0].flatten()}

        """
        Now we have all the detections done, we need to whittle down to correspondences.
            - Find the intersection of used frames
            - For each used frame find the intersection of detected points.
                - The indices that show up here are ones we can add to a list for both cameras that will hold the point pairs

            - The problem is that used_both gives an index error - lists are the wrong idea here
        """

        used_c1 = set(used_c1)
        used_c2 = set(used_c2)

        used_both = used_c1.intersection(used_c2)

        for indx in used_both:
            c1_pts = all_pts_c1[indx]
            c2_pts = all_pts_c2[indx]
            ids_both = set(c1_pts.keys()).intersection(set(c2_pts.keys()))

            cam_1_im_pts.append(
                all_pts_c1[indx][ids_both]
                )
            
            cam_2_im_pts.append(
                all_pts_c2[indx][ids_both]
            )

        retval, mask = cv.findEssentialMat(cam_1_im_pts,cam_2_im_pts,cameraMatrix1)

        # Decomposing returns possible values Rposs1, Rposs2, tposs
        return cv.decomposeEssentialMat(retval)

    def visually_inspect(self,frame,obj_pts,rvec,tvec,cam_mat,dist,err,frame_num,rms_error):
        # used is a vector that tells us whether any points at all were found and thus whether the ith image was used
        # the jth element of per_view_errors corresponds to the used[j]th element of the frame_generator
        im_pts = cv.projectPoints(obj_pts,rvec,tvec,cam_mat,dist)

        framergb = cv.cvtColor(frame,cv.COLOR_GRAY2BGR)

        cv.drawChessboardCorners(framergb,(10,10),im_pts[0],True)
        cv.putText(framergb,f"Error: {err[0]:.1f}px",(10,30),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        cv.putText(framergb,f"Frame: {frame_num}",(10,70),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        cv.putText(framergb,f"RMS Error (all frames): {rms_error:.2f}",(10,110),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        cv.imshow('Visual inspection of camera calibration',framergb)
        cv.waitKey()

    def __repr__(self):
        return f"RMS Error: {self.retval:.2f}px"