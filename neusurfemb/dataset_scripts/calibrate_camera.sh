#!/bin/bash
CALIBRATION_VIDEO_PATH=$1;
OUTPUT_FOLDER_PATH=$2;
CALIBRATION_CFG_FILENAME=$3;
SHOW_REPORT=$4; # Default: True.

if [ -z "${CALIBRATION_VIDEO_PATH}" ]
then
    echo "Please provide the path to the calibration video.";
    exit 1;
else
    CALIBRATION_VIDEO_PATH=$(realpath ${CALIBRATION_VIDEO_PATH});
    if [[ ! -e "${CALIBRATION_VIDEO_PATH}" ]]; then
        echo "Could not find a calibration video at '${CALIBRATION_VIDEO_PATH}'.";
        exit 1;
    fi
fi
if [ -z "${OUTPUT_FOLDER_PATH}" ]
then
    echo "Please provide the path to the folder where to store the calibration output.";
    exit 1;
else
    OUTPUT_FOLDER_PATH=$(realpath ${OUTPUT_FOLDER_PATH});
fi
if [ -z "${CALIBRATION_CFG_FILENAME}" ]
then
    echo "Filename of the calibration config file was not provided. Assuming A3 paper format ('april_6x6_A3.yaml').";
    CALIBRATION_CFG_FILENAME="april_6x6_A3.yaml";
fi
if [ -z "${SHOW_REPORT}" ]
then
    # Default: True.
    SHOW_REPORT_STR="";
else
    # Convert to lowercase.
    SHOW_REPORT=${SHOW_REPORT,,};
    if [ ${SHOW_REPORT} = "true" ]
    then
        SHOW_REPORT_STR="";
    elif [ ${SHOW_REPORT} = "false" ]
    then
        SHOW_REPORT_STR="--dont-show-report";
    else
        echo "Invalid value '${SHOW_REPORT}' for 'SHOW_REPORT'. Valid values are: 'true', 'false' (case insensitive).";
        exit 1;
    fi
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
CALIBRATION_VIDEO_FILENAME=$(basename ${CALIBRATION_VIDEO_PATH});
CALIBRATION_VIDEO_FILENAME=${CALIBRATION_VIDEO_FILENAME//.mp4/};
pushd ${CATKIN_WS}/src/video2bag;
python convert_video_to_bag.py \
    ${CALIBRATION_VIDEO_PATH} \
    --output_file ${OUTPUT_FOLDER_PATH}/${CALIBRATION_VIDEO_FILENAME}.bag \
    --output_dir ${OUTPUT_FOLDER_PATH} \
    --do_not_save_images \
    --sleep_rate 0.2 \
    --div_num 20;
rosbag reindex ${OUTPUT_FOLDER_PATH}/${CALIBRATION_VIDEO_FILENAME}.bag && rm ${OUTPUT_FOLDER_PATH}/${CALIBRATION_VIDEO_FILENAME}.orig.bag;
# Run calibration with `kalibr`. Make sure to use the configuration file that
# matches the printed paper format for the `--target` argument:
roscd kalibr;
python python/kalibr_calibrate_cameras \
    ${SHOW_REPORT_STR} \
    --models pinhole-radtan \
    --topics /camera/image_raw \
    --bag ${OUTPUT_FOLDER_PATH}/${CALIBRATION_VIDEO_FILENAME}.bag \
    --target ${NEUSURFEMB_ROOT}/cfg/calibration/${CALIBRATION_CFG_FILENAME};
# Remove temporary bag file.
rm ${OUTPUT_FOLDER_PATH}/${CALIBRATION_VIDEO_FILENAME}.bag;