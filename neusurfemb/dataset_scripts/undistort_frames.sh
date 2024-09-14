#!/bin/bash
VIDEO_PATH=$1;
CALIBRATION_FOLDER_PATH=$2;
OUTPUT_FOLDER_PATH=$3;

if [ -z "${VIDEO_PATH}" ]
then
    echo "Please provide the path to the video of the object.";
    exit 1;
else
    VIDEO_PATH=$(realpath ${VIDEO_PATH});
    if [[ ! -e "${VIDEO_PATH}" ]]; then
        echo "Could not find a video at '${VIDEO_PATH}'.";
        exit 1;
    fi
fi
if [ -z "${CALIBRATION_FOLDER_PATH}" ]
then
    echo "Please provide the path to the folder with the calibration results.";
    exit 1;
else
    CALIBRATION_FOLDER_PATH=$(realpath ${CALIBRATION_FOLDER_PATH});
fi
if [ -z "${OUTPUT_FOLDER_PATH}" ]
then
    echo "Please provide the path to the folder where to store the undistorted images.";
    exit 1;
else
    OUTPUT_FOLDER_PATH=$(realpath ${OUTPUT_FOLDER_PATH});
fi

if [ -z "${CATKIN_WS}" ]
then
    echo "Make sure the environmental variable 'CATKIN_WS' is defined.";
    exit 1;
fi
if [ -z "${NEUSURFEMB_VIRTUALENV}" ]
then
    echo "Make sure the environmental variable 'NEUSURFEMB_VIRTUALENV' is defined.";
    exit 1;
fi
if [ -z "${NEUSURFEMB_ROOT}" ]
then
    echo "Make sure the environmental variable 'NEUSURFEMB_ROOT' is defined.";
    exit 1;
fi

source /opt/ros/noetic/setup.bash;
source ${CATKIN_WS}/devel/setup.bash;
source ${NEUSURFEMB_VIRTUALENV}/bin/activate;
# Convert the video to a rosbag:
VIDEO_FILENAME=$(basename ${VIDEO_PATH});
VIDEO_FILENAME=${VIDEO_FILENAME//.mp4/};
pushd ${CATKIN_WS}/src/video2bag;
BAG_PATH=${OUTPUT_FOLDER_PATH}/${VIDEO_FILENAME}.bag;
python convert_video_to_bag.py \
    ${VIDEO_PATH} \
    --output_file ${BAG_PATH} \
    --output_dir ${OUTPUT_FOLDER_PATH} \
    --sleep_rate 0.2 \
    --div_num 40 \
    --calibration_file ${CALIBRATION_FOLDER_PATH}/*camchain.yaml;
rosbag reindex ${BAG_PATH} && rm ${OUTPUT_FOLDER_PATH}/${VIDEO_FILENAME}.orig.bag;
# Undistort images.
UNDISTORTED_BAG_PATH=${OUTPUT_FOLDER_PATH}/${VIDEO_FILENAME}_undistorted.bag;
cd ${CATKIN_WS}/src;
roslaunch undistort_object_video.launch \
    bag_path:=${BAG_PATH} \
    output_path:=${UNDISTORTED_BAG_PATH};
# Extract the images as a dataset.
UNDISTORTED_BAG_PATH=${UNDISTORTED_BAG_PATH//.bag/}*.bag;
UNDISTORTED_OUTPUT_FOLDER=${OUTPUT_FOLDER_PATH}/dataset_version/;
python ${NEUSURFEMB_ROOT}/neusurfemb/dataset_scripts/extract_dataset_from_bag.py \
    --bag-path ${UNDISTORTED_BAG_PATH} \
    --output-folder ${UNDISTORTED_OUTPUT_FOLDER} \
    --freq 1;
# Remove temporary files.
rm -r ${OUTPUT_FOLDER_PATH}/*.bag ${OUTPUT_FOLDER_PATH}/images;