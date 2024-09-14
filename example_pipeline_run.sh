#!/bin/bash
set -e;

NEUSURFEMB_ROOT=$1;
EXAMPLE_DATA_FOLDER=$2;
CALIBRATION_VIDEO_FILENAME=$3;
OBJECT_VIDEO_FILENAME=$4;
EVALUATION_DATASET_ZIP_FILENAME=$5;
RENDERER_POSE_ESTIMATION_EVALUATION=$6;

if [ -z "${NEUSURFEMB_ROOT}" ]
then
    echo "Please provide the path to the root of the repo.";
    exit 1;
else
    NEUSURFEMB_ROOT=$(realpath ${NEUSURFEMB_ROOT});
fi
if [ -z "${EXAMPLE_DATA_FOLDER}" ]
then
    echo "Please provide the path to the folder containing the example data.";
    exit 1;
else
    EXAMPLE_DATA_FOLDER=$(realpath ${EXAMPLE_DATA_FOLDER});
fi
if [ -z "${CALIBRATION_VIDEO_FILENAME}" ]
then
    echo "Please provide the filename of the calibration video in '${EXAMPLE_DATA_FOLDER}'.";
    exit 1;
else
    CALIBRATION_VIDEO_PATH=${EXAMPLE_DATA_FOLDER}/${CALIBRATION_VIDEO_FILENAME};
    if [[ ! -e "${CALIBRATION_VIDEO_PATH}" ]]; then
        echo "Could not find a calibration video at '${CALIBRATION_VIDEO_PATH}'.";
        exit 1;
    fi
fi
if [ -z "${OBJECT_VIDEO_FILENAME}" ]
then
    echo "Please provide the filename of the object video in '${EXAMPLE_DATA_FOLDER}'.";
    exit 1;
else
    OBJECT_VIDEO_PATH=${EXAMPLE_DATA_FOLDER}/${OBJECT_VIDEO_FILENAME};
    if [[ ! -e "${OBJECT_VIDEO_PATH}" ]]; then
        echo "Could not find an object video at '${OBJECT_VIDEO_PATH}'.";
        exit 1;
    fi
fi
if [ -z "${EVALUATION_DATASET_ZIP_FILENAME}" ]
then
    echo "Please provide the filename of the evaluation dataset archive in '${EVALUATION_DATASET_ZIP_FILENAME}'.";
    exit 1;
else
    EVALUATION_DATASET_ZIP_PATH=${EXAMPLE_DATA_FOLDER}/${EVALUATION_DATASET_ZIP_FILENAME};
    if [[ ! -e "${EVALUATION_DATASET_ZIP_PATH}" ]]; then
        echo "Could not find a zip archive for the evaluation dataset at '${EVALUATION_DATASET_ZIP_PATH}'.";
        exit 1;
    else
        # - Create a temporary folder to extract the archive.
        temp_dir=$(mktemp -d);
        unzip -q "$EVALUATION_DATASET_ZIP_PATH" -d "$temp_dir";
        # - Check if the extracted archive contains exactly one folder at its
        #   top level.
        EXTRACTED_ITEMS=$(ls -1 "$temp_dir");
        EXTRACTED_ITEMS_COUNT=$(echo "$EXTRACTED_ITEMS" | wc -l);
        if [ "$EXTRACTED_ITEMS_COUNT" -ne 1 ]; then
            echo "The archive does not contain exactly one top-level folder.";
            rm -rf "$temp_dir";
            exit 1;
        fi
        # - Find the name of the folder and check if it is indeed a folder.
        EXTRACTED_FOLDER_NAME="$EXTRACTED_ITEMS";
        FULL_EXTRACTED_PATH="$temp_dir/$EXTRACTED_FOLDER_NAME";
        if [ ! -d "$FULL_EXTRACTED_PATH" ]; then
            echo "The top-level item is not a folder.";
            rm -rf "$temp_dir";
            exit 1;
        fi
        # - Move the extracted folder to the location of the zip file.
        ZIP_DIR=$(dirname "$EVALUATION_DATASET_ZIP_PATH");
        mv "$FULL_EXTRACTED_PATH" "$ZIP_DIR";
        EVALUATION_DATASET_FOLDER="/home/data/${EXTRACTED_FOLDER_NAME}";
        # - Clean up the temporary directory.
        rm -rf "$temp_dir";
    fi
fi
if [ -z "${RENDERER_POSE_ESTIMATION_EVALUATION}" ]
then
    echo "Please provide the type of renderer you would like to use for pose estimation on the evaluation dataset (either 'moderngl' or 'neus2').";
    exit 1;
else
    if [[ "${RENDERER_POSE_ESTIMATION_EVALUATION}" != "moderngl" && "${RENDERER_POSE_ESTIMATION_EVALUATION}" != "neus2" ]]; then
        echo "Valid values for the type of renderer for pose estimation on the evaluation dataset are 'moderngl' and 'neus2'. Found '${RENDERER_POSE_ESTIMATION_EVALUATION}'."
        exit 1
    fi
fi


# 1. Run camera calibration.
echo -e "\033[0;33m*** 1. Running camera calibration ***\033[0m";
SHOW_REPORT=false;
${NEUSURFEMB_ROOT}/docker/run_docker.sh -d ${EXAMPLE_DATA_FOLDER} \
    bash \
    /home/src/neusurfemb/neusurfemb/dataset_scripts/calibrate_camera.sh \
    /home/data/${CALIBRATION_VIDEO_FILENAME} \
    /home/data \
    april_6x6_A0.yaml \
    ${SHOW_REPORT};

# 2. Extract and undistort object frames.
echo -e "\033[0;33m*** 2. Extracting and undistorting object frames ***\033[0m";
OUTPUT_FOLDER_NAME=${OBJECT_VIDEO_FILENAME//.*/};
OUTPUT_FOLDER_PATH=/home/data/${OUTPUT_FOLDER_NAME};
${NEUSURFEMB_ROOT}/docker/run_docker.sh -d ${EXAMPLE_DATA_FOLDER} \
    bash \
    /home/src/neusurfemb/neusurfemb/dataset_scripts/undistort_frames.sh \
    /home/data/${OBJECT_VIDEO_FILENAME} \
    /home/data \
    ${OUTPUT_FOLDER_PATH};

# 3. Pose labeling (SfM).
echo -e "\033[0;33m*** 3. Running pose labeling (SfM) ***\033[0m";
DATASET_FOLDER=${OUTPUT_FOLDER_PATH}/dataset_version;
${NEUSURFEMB_ROOT}/docker/run_docker.sh -d ${EXAMPLE_DATA_FOLDER} \
    python \
    /home/src/neusurfemb/neusurfemb/dataset_scripts/pose_labeling_given_intrinsics.py \
    ${DATASET_FOLDER};

# 4. Extract object masks.
echo -e "\033[0;33m*** 4. Extracting object masks ***\033[0m";
${NEUSURFEMB_ROOT}/docker/run_docker.sh -d ${EXAMPLE_DATA_FOLDER} -- \
    python \
    /home/src/neusurfemb/neusurfemb/dataset_scripts/extract_object_masks_bbox.py \
    --dataset-folder ${DATASET_FOLDER};

# 5. Convert to NeuS2 dataset.
echo -e "\033[0;33m*** 5. Converting to NeuS2 dataset ***\033[0m";
${NEUSURFEMB_ROOT}/docker/run_docker.sh -d ${EXAMPLE_DATA_FOLDER} -- \
    python \
    /home/src/neusurfemb/neusurfemb/dataset_scripts/real_dataset_to_neus.py \
    --dataset-folder ${DATASET_FOLDER};

# 6. Adapt dataset scale.
echo -e "\033[0;33m*** 6. Adapting dataset scale ***\033[0m";
BOUND_EXTENT=0.55;
${NEUSURFEMB_ROOT}/docker/run_docker.sh -d ${EXAMPLE_DATA_FOLDER} -- \
    python \
    /home/src/neusurfemb/neusurfemb/dataset_scripts/adapt_dataset_scale.py \
    --scene-transform-path ${DATASET_FOLDER}/transforms.json \
    --name bound_${BOUND_EXTENT} \
    --n-steps 4000 \
    --should-convert-to-mm \
    --bound-extent ${BOUND_EXTENT};
DATASET_FOLDER=${DATASET_FOLDER}/neus_rescaling/bound_${BOUND_EXTENT}/new_dataset;

# 7. Train NeuS2 and generate dataset.
echo -e "\033[0;33m*** 7. Training NeuS2 and generating dataset ***\033[0m";
TRAININGS_FOLDER=/home/data/trainings;
EXPERIMENT_FOLDER=${TRAININGS_FOLDER}/${OBJECT_VIDEO_FILENAME//.*/};
NUM_TRAIN_ITERS=20000;
COMPUTE_ORIENTED_BBOX=true;
${NEUSURFEMB_ROOT}/docker/run_docker.sh -d ${EXAMPLE_DATA_FOLDER} -- \
    "mkdir -p ${TRAININGS_FOLDER} && \
    python \
    /home/src/neusurfemb/training_scripts/train_neus.py \
    ${DATASET_FOLDER} \
    --workspace ${EXPERIMENT_FOLDER} \
    --num-iters ${NUM_TRAIN_ITERS} \
    --compute-oriented-bounding-box ${COMPUTE_ORIENTED_BBOX} \
    --synthetic-data-config-path /home/src/neusurfemb/cfg/syn_data_generation/config_10000_views.yml";

# 8. Train SurfEmb (Option 2 here).
echo -e "\033[0;33m*** 8. Training SurfEmb ***\033[0m";
EXPERIMENT_BASE_NAME=$(basename ${EXPERIMENT_FOLDER});
AUGMENTATION_DATASET_FOLDER=${TRAININGS_FOLDER}/augmentation_datasets;
MODELS_FOLDER=${TRAININGS_FOLDER}/models;
LOGS_FOLDER=${TRAININGS_FOLDER}/logs;
BOP_FOLDER=${TRAININGS_FOLDER}/bop;
# - Create folders to be later symlinked to store data and training outputs.
${NEUSURFEMB_ROOT}/docker/run_docker.sh -d ${EXAMPLE_DATA_FOLDER} -- \
    "mkdir -p ${AUGMENTATION_DATASET_FOLDER} && \
    mkdir -p ${MODELS_FOLDER} && \
    mkdir -p ${LOGS_FOLDER}";
# - Create symlinks for training and train.
${NEUSURFEMB_ROOT}/docker/run_docker.sh -d ${EXAMPLE_DATA_FOLDER} -- \
    "export WANDB_MODE=disabled && \
    cd /home/src/neusurfemb/third_party/surfemb && \
    ln -s ${AUGMENTATION_DATASET_FOLDER} data/augmentation_datasets && \
    ln -s ${MODELS_FOLDER} data/models && \
    ln -s ${LOGS_FOLDER} data/logs && \
    mkdir -p data/bop && \
    ln -s ${EXPERIMENT_FOLDER}/data_for_surfemb/ data/bop/${EXPERIMENT_BASE_NAME} && \
    ln -s ${EXPERIMENT_FOLDER}/data_for_surfemb/surface_samples/ data/surface_samples/${EXPERIMENT_BASE_NAME} && \
    ln -s ${EXPERIMENT_FOLDER}/data_for_surfemb/surface_samples_normals/ data/surface_samples_normals/${EXPERIMENT_BASE_NAME} && \
    python -m surfemb.scripts.train \
    ${EXPERIMENT_BASE_NAME} \
    --gpus 0 \
    --neus2-dataset \
    --renderer-type moderngl";

# 9. Train YOLO.
echo -e "\033[0;33m*** 9. Training YOLO ***\033[0m";
OBJECT_NAME="helmet";
${NEUSURFEMB_ROOT}/docker/run_docker.sh -d ${EXAMPLE_DATA_FOLDER} -- \
    python \
    /home/src/neusurfemb/training_scripts/train_yolo.py \
    --dataset-folder ${DATASET_FOLDER} \
    --object-name ${OBJECT_NAME};

# 10. Evaluate on the evaluation dataset.
echo -e "\033[0;33m*** 10. Evaluating on the evaluation dataset ***\033[0m";
# - Generate pose estimation configuration file based on the example
#   `pose_estimation_moderngl_renderer.yml`. These commands simply set the
#   correct values for the path to the trained checkpoint (hack: the checkpoint
#   name is first written to a temporary file) and the path to the intrinsics of
#   the evaluation dataset. You may also do so manually.
CKPT_NAME_FILE=${OUTPUT_FOLDER_PATH}/ckpt_name;
if [[ "${RENDERER_POSE_ESTIMATION_EVALUATION}" == "moderngl" ]]; then
    POSE_EST_CFG_FILE=${EXPERIMENT_FOLDER}/pose_estimation_moderngl_renderer.yml;
    ${NEUSURFEMB_ROOT}/docker/run_docker.sh -d ${EXAMPLE_DATA_FOLDER} -- \
        "find ${MODELS_FOLDER} -type f -iname \"*.ckpt\" -fprint0 ${CKPT_NAME_FILE} && \
        sed -i \"s/\\\x0//g\" ${CKPT_NAME_FILE} && \
        (cat ${CKPT_NAME_FILE} | xargs -I {} cp {} ${EXPERIMENT_FOLDER}/checkpoints/ ) && \
        sed -i \"s|${MODELS_FOLDER}\/|checkpoints\/|g\" ${CKPT_NAME_FILE} && \
        sed -i \"s/^/model_path: \\\"""/\" ${CKPT_NAME_FILE} && \
        echo \\\" >> ${CKPT_NAME_FILE} && \
        cp /home/src/neusurfemb/cfg/evaluation/pose_estimation_moderngl_renderer.yml ${EXPERIMENT_FOLDER}/ &&
        sed -e \"/model_path:.*/ {\" -e \"r ${CKPT_NAME_FILE}\" -e \"d\" -e \"}\" -i ${POSE_EST_CFG_FILE} && \
        (sed -i 's|intrinsics_path: \"|intrinsics_path: \"'${EVALUATION_DATASET_FOLDER}\/'|' ${POSE_EST_CFG_FILE}) && \
        rm ${CKPT_NAME_FILE}";
elif [[ "${RENDERER_POSE_ESTIMATION_EVALUATION}" == "neus2" ]]; then
    POSE_EST_CFG_FILE=${EXPERIMENT_FOLDER}/pose_estimation_neus2_renderer.yml;
    ${NEUSURFEMB_ROOT}/docker/run_docker.sh -d ${EXAMPLE_DATA_FOLDER} -- \
        "find ${MODELS_FOLDER} -type f -iname \"*.ckpt\" -fprint0 ${CKPT_NAME_FILE} && \
        sed -i \"s/\\\x0//g\" ${CKPT_NAME_FILE} && \
        (cat ${CKPT_NAME_FILE} | xargs -I {} cp {} ${EXPERIMENT_FOLDER}/checkpoints/ ) && \
        sed -i \"s|${MODELS_FOLDER}\/|checkpoints\/|g\" ${CKPT_NAME_FILE} && \
        sed -i \"s/^/model_path: \\\"""/\" ${CKPT_NAME_FILE} && \
        echo \\\" >> ${CKPT_NAME_FILE} && \
        cp /home/src/neusurfemb/cfg/evaluation/pose_estimation_neus2_renderer.yml ${EXPERIMENT_FOLDER}/ &&
        sed -e \"/model_path:.*/ {\" -e \"r ${CKPT_NAME_FILE}\" -e \"d\" -e \"}\" -i ${POSE_EST_CFG_FILE} && \
        (sed -i 's|intrinsics_path: \"|intrinsics_path: \"'${EVALUATION_DATASET_FOLDER}\/'|' ${POSE_EST_CFG_FILE}) && \
        (sed -i '/neus2_checkpoint_folders:/!b;n;c\  - "\"${EXPERIMENT_FOLDER}/checkpoints/\""' ${POSE_EST_CFG_FILE}) && \
        rm ${CKPT_NAME_FILE}";
fi
# - Run evaluation script.
${NEUSURFEMB_ROOT}/docker/run_docker.sh -d ${EXAMPLE_DATA_FOLDER} -- \
    python \
    /home/src/neusurfemb/test_pose_estimator.py \
    --image-folder ${EVALUATION_DATASET_FOLDER}/rgb/ \
    --pose-estimation-cfg-file ${POSE_EST_CFG_FILE} \
    --bbox-folder ${EVALUATION_DATASET_FOLDER}/bbox/;
