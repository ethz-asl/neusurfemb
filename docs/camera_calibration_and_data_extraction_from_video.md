# Camera calibration and data extraction from video
In the following, we provide instructions on how to extract undistorted images from a video recorded with a smartphone, to be used to build a NeuS2-based object model.
As mentioned in the main README, this step is not strictly necessary, as the SfM module can also process raw images and concurrently estimate the camera intrinsics, but in our experiments pre-calibrating the camera and providing undistorted images to the SfM module resulted in higher-quality NeuS2 reconstruction.

Our calibration solution is based on the ROS-dependent packages [`kalibr`](https://github.com/ethz-asl/kalibr) and [`image_undistort`](https://github.com/ethz-asl/image_undistort). For this purpose, it is here assumed that a ROS Noetic catkin workspace `${CATKIN_WS}` was previously set up and built according to our repo [installation instructions](./install.md). Note that if you are using a Docker installation through the provided Dockerfile, you will find the already catkin workspace in the folder `/home/src/neusurfemb/ros/catkin_ws` (hence `CATKIN_WS=/home/src/neusurfemb/ros/catkin_ws`).
The path to the root of the `neusurfemb` installation virtualenv and to the root of this repo are assumed to be stored respectively in the `${NEUSURFEMB_VIRTUALENV}` and in the `${NEUSURFEMB_ROOT}` environmental variables, which correspond respectively to `/home/src/.virtualenvs/neusurfemb` and `/home/src/neusurfemb` when using a Docker installation through the provided Dockerfile (cf. installation instructions).

## Camera calibration
To extract undistorted frames, it is necessary to calibrate the smartphone camera used to record the video. This step only needs to be performed once: Once the camera is calibrated, recordings can be made for multiple new objects and the same camera intrinsics can be used, assuming that the same camera and lens are used across all the recordings.

### Record a video for calibration
Using the same smartphone camera and lens that you will later use to capture the video to construct the object model, record a video with the camera pointing at a calibration pattern compatible with [`kalibr`](https://github.com/ethz-asl/kalibr).
We provide a set of PDF files (and associated configuration files) of calibration patterns made of grids of April tags in [this folder](../cfg/calibration/). The patterns were generated according to the specifications in the [`kalibr` wiki](https://github.com/ethz-asl/kalibr/wiki/calibration-targets#a-aprilgrid) and are meant to be printed on different formats (A0, A3, or A4, as indicated in the filenames). We recommend using A3 format, or even better A0 format, to improve calibration results. Make sure to print the patterns at 100% scale and keep them on a flat, rigid surface throughout the calibration process.

<details>
<summary>How to generate custom patterns, if necessary.</summary>

As a first step, make sure to install `texlive` (not necessary if using the Docker-based installation):
```bash
apt update && apt install -y texlive;
```

Then use the following instructions, which will each produce a `target.pdf` file:
```bash
source ${CATKIN_WS}/devel/setup.bash;
source ${NEUSURFEMB_VIRTUALENV}/bin/activate;
roscd kalibr;
# A0.
python python/kalibr_create_target_pdf --type apriltag --nx 6 --ny 6 --tsize 0.083 --tspace 0.3;
# A3.
python python/kalibr_create_target_pdf --type apriltag --nx 6 --ny 6 --tsize 0.029 --tspace 0.3;
# A4.
python python/kalibr_create_target_pdf --type apriltag --nx 6 --ny 6 --tsize 0.021 --tspace 0.3;
```
</details>

When recording the video, keep the calibration pattern static, and slowly move your camera so as to cover all sides of the image plane, while keeping as many April tags visible as possible at all times. Rotate and translate the camera along all axes, including in particular also "side" views of the calibration pattern, with large perspective distortion. You may check [this video](https://www.youtube.com/watch?v=puNXsnrYWTY) as a reference for the type camera movements to use.
We recommend recording with Full HD quality if available and in landscape mode, since we observed that `kalibr` can fail to process portrait videos. In case you record the video in portrait mode, make sure to rotate the video so that it is in landscape mode, using, _e.g._, `ffmpeg -i <PORTAIT_CALIBRATION_VIDEO_FILENAME>.mp4 -metadata:s:v:0 rotate=180 -codec copy <LANDSCAPE_CALIBRATION_VIDEO_FILENAME>.mp4`.

### Calibrate the camera using the recorded video
We provide the script [`calibrate_camera.sh`](../neusurfemb/dataset_scripts/calibrate_camera.sh) to run the camera calibration using the recorded video. You may run it as follows:
```bash
bash ${NEUSURFEMB_ROOT}/neusurfemb/dataset_scripts/calibrate_camera.sh <CALIBRATION_VIDEO_PATH> <OUTPUT_FOLDER_PATH> [<CALIBRATION_CFG_FILENAME>];
```
where
- `<CALIBRATION_VIDEO_PATH>` is the path to the calibration video;
- `<OUTPUT_FOLDER_PATH>` is the path to the folder where the calibration results will be stored;
- `<CALIBRATION_CFG_FILENAME>` is the filename of the calibration config file, relative to the [`cfg/calibration` folder of the repo](../cfg/calibration/). The default is `'april_6x6_A3.yaml'`.

If you are using the Docker installation, place the calibration video in a folder `${SHARED_HOST_FOLDER}` of your choice on your host computer, which will be mounted for read-write at the location `/home/data` in the Docker container. Then, use the [`run_docker.sh` scripts](../docker/run_docker.sh) to run the above command as follows:
```bash
${NEUSURFEMB_ROOT}/docker/run_docker.sh -d ${SHARED_HOST_FOLDER} bash /home/src/neusurfemb/neusurfemb/dataset_scripts/calibrate_camera.sh <CALIBRATION_VIDEO_PATH> <OUTPUT_FOLDER_PATH> [<CALIBRATION_CFG_FILENAME>];
```
Note that both `<CALIBRATION_VIDEO_PATH>` and `<OUTPUT_FOLDER_PATH>` are paths in the Docker container; in order to access files from the host computer, both paths should be below the `/home/data` folder of the container, and the corresponding paths on the host computer should be in the folder `${SHARED_HOST_FOLDER}`. For instance, one can run `./run_docker.sh -d ${SHARED_HOST_FOLDER} bash /home/src/neusurfemb/neusurfemb/dataset_scripts/calibrate_camera.sh /home/data/calibration/calib_video.mp4 /home/data/calibration/calibration_output` to process a `calib_video.mp4` video in the host folder `${SHARED_HOST_FOLDER}/calibration` and have the output stored in the host folder `${SHARED_HOST_FOLDER}/calibration/calibration_output`.

The calibration process will display a GUI output as well as an output on the terminal. A good calibration should have a reprojection error variance of around 0.25 pixels or less.

## Extract and undistort frames to later construct a NeuS2 object model
Once the camera has been calibrated, record a video of the object for which you would like to train a pose estimator, following the recommendations in the [main README](../README.md#data-recording-and-extraction-skip-for-bop-datasets). Make sure to use the same camera lens and video orientation as those used for calibration.
Then, you can extract frames from the recorded video and pre-process them by undistorting them using the camera calibration. To do so, we provide the script [`undistort_frames.sh`](../neusurfemb/dataset_scripts/undistort_frames.sh), which you may run as follows:
```bash
bash ${NEUSURFEMB_ROOT}/neusurfemb/dataset_scripts/undistort_frames.sh <VIDEO_PATH> <CALIBRATION_FOLDER_PATH> <OUTPUT_FOLDER_PATH>;
```
where
- `<VIDEO_PATH>` is the path to the video of the object;
- `<CALIBRATION_FOLDER_PATH>` is the path to the folder containing the calibration results from the above calibration steps;
- `<OUTPUT_FOLDER_PATH>` is the folder that will store the output of the extraction and undistortion process. The output of the process, in the correct format to be processed in later stages, will be saved in the `<OUTPUT_FOLDER_PATH>/dataset_version` subfolder.

If necessary, adjust the parameters for the flags `--sleep_rate`, `--div_num`, and `--freq` in the `undistort_frames.sh` script.
You can ignore the following two errors: `Optimal output camera parameters cannot be set until the input camera parameters have been given` and `Image frame name is blank, cannot construct tf`.

If using the Docker installation, adapt the command similarly to what done above for the calibration step:
```bash
${NEUSURFEMB_ROOT}/docker/run_docker.sh -d ${SHARED_HOST_FOLDER} bash /home/src/neusurfemb/neusurfemb/dataset_scripts/undistort_frames.sh <VIDEO_PATH> <CALIBRATION_FOLDER_PATH> <OUTPUT_FOLDER_PATH>;
```
for instance `${NEUSURFEMB_ROOT}/docker/run_docker.sh -d ${SHARED_HOST_FOLDER} bash /home/src/neusurfemb/neusurfemb/dataset_scripts/undistort_frames.sh /home/data/object1/object1.mp4 /home/data/calibration/calibration_output /home/data/object1` to extract and undistort frames from the video `${SHARED_HOST_FOLDER}/object1/object1.mp4` with the calibration output already stored in `${SHARED_HOST_FOLDER}/calibration/calibration_output` and the processed dataset being saved to `${SHARED_HOST_FOLDER}/object1/dataset_version` (all paths on the host computer).

### [Optional] Filter frames based on sharpness
Optionally, you may remove frames that are too blurry using the script [`filter_frames_based_on_sharpness.py`](../neusurfemb/dataset_scripts/filter_frames_based_on_sharpness.py):
```bash
python ${NEUSURFEMB_ROOT}/neusurfemb/dataset_scripts/filter_frames_based_on_sharpness.py --dataset-folder <OUTPUT_FOLDER_PATH>/dataset_version --min-valid-sharpness <MIN_VALID_SHARPNESS>;
```
where `<MIN_VALID_SHARPNESS>` is a threshold to be manually set (we find values between 150 and 500 to work well depending on the dataset quality).

Docker equivalent:
```bash
${NEUSURFEMB_ROOT}/docker/run_docker.sh -d ${SHARED_HOST_FOLDER} python /home/src/neusurfemb/neusurfemb/dataset_scripts/filter_frames_based_on_sharpness.py --dataset-folder <OUTPUT_FOLDER_PATH>/dataset_version --min-valid-sharpness <MIN_VALID_SHARPNESS>;
```