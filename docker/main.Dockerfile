FROM nvidia/cuda:11.8.0-devel-ubuntu20.04 AS ros_setup

SHELL ["/bin/bash", "-c"] 

# Install ROS first. From https://github.com/ethz-asl/moma/blob/master/docker/dev_cuda.Dockerfile.
RUN apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata 

# - We're in Zurich!
ENV TZ="Europe/Zurich"

# - Install packages.
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2

# - Set up sources.list.
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list

# - Set up keys.
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# - Set up environment.
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ENV ROS_DISTRO=noetic

# - Install ROS.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full

# - Update Ubuntu packages to latest.
RUN apt-get -qq update && apt-get -qq upgrade

# - Update git and set to always point to https.
RUN apt-get -qq update && apt-get install -y curl apt-utils
RUN apt-get -qq update && apt-get install -y git git-lfs
# - Install tools.
RUN apt-get -qq update && apt-get install -y python3-catkin-tools python3-vcstool python3-pip python-is-python3
# - Install other system deps.
RUN apt-get -qq update && apt-get install -y qtbase5-dev bash-completion
# - Clear cache to keep layer size down
RUN rm -rf /var/lib/apt/lists/*

# - Clone the NeuSurfEmb repo and its submodules.
ENV SRC_HOME=/home/src
RUN mkdir ${SRC_HOME}
WORKDIR ${SRC_HOME}
RUN git clone https://github.com/ethz-asl/neusurfemb.git
ENV NEUSURFEMB_ROOT=${SRC_HOME}/neusurfemb
WORKDIR ${NEUSURFEMB_ROOT}
RUN git fetch && git checkout main

RUN git submodule update --init --recursive
# - Set up catkin workspace.
ENV CATKIN_WS=${NEUSURFEMB_ROOT}/ros/catkin_ws
WORKDIR ${CATKIN_WS}
RUN catkin init && catkin config --extend /opt/ros/noetic --cmake-args -DCMAKE_BUILD_TYPE=Release
WORKDIR ${CATKIN_WS}/src/catkin_simple
RUN git checkout 0e62848b12da76c8cc58a1add42b4f894d1ac21e && git submodule update --init --recursive
WORKDIR ${CATKIN_WS}/src/kalibr
RUN git checkout 3c2856c77bc1ec22a5d02d49a0e889df0356cb17 && git submodule update --init --recursive
WORKDIR ${CATKIN_WS}/src/image_undistort
RUN git checkout a8fa9cdcf2c25f8ca83556b7bfb891fe64a3612b && git submodule update --init --recursive
WORKDIR ${CATKIN_WS}/src
RUN apt update && apt install -y libv4l-dev libsuitesparse-dev
RUN catkin build -j12
RUN source ${CATKIN_WS}/devel/setup.sh
# - Install TeX (required to generate patterns for calibration).
RUN apt update && apt install -y texlive

# Set Python environment.
# - Create a virtualenv:
RUN apt update && apt -y upgrade && apt install -y python3.8-venv
ENV NEUSURFEMB_VIRTUALENV=${SRC_HOME}/.virtualenvs/neusurfemb
RUN mkdir -p ${NEUSURFEMB_VIRTUALENV}
RUN python3 -m venv ${NEUSURFEMB_VIRTUALENV}
CMD source ${NEUSURFEMB_VIRTUALENV}/bin/activate
ENV PATH="${NEUSURFEMB_VIRTUALENV}/bin:$PATH"
RUN pip install --upgrade pip
# - Install pip dependencies for kalibr and image_undistort:
RUN pip install pyx~=0.16 empy~=4.1
# - Install NeuS2:
#   - Upgrade CMake.
RUN apt-get update \
  && apt-get -y install build-essential \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/* \
  && wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-Linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /opt/cmake-3.24.1 \
      && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.24.1 \
      && rm /tmp/cmake-install.sh \
      && ln -s /opt/cmake-3.24.1/bin/* /usr/local/bin
#   - Build NeuS2.
WORKDIR ${NEUSURFEMB_ROOT}/third_party/NeuS2
RUN git checkout 01f322a6701a762564e3bc250ac561bdd7b7659e && git submodule update --init --recursive
RUN cmake . -B build
RUN cmake --build build --config RelWithDebInfo -j
RUN cp ${NEUSURFEMB_ROOT}/third_party/NeuS2/build/pyngp.cpython-38-x86_64-linux-gnu.so \
    ${NEUSURFEMB_VIRTUALENV}/lib/python3.8/site-packages/
# - Install dependencies of NeuS (here assuming CUDA Toolkit 11.8 is installed):
RUN pip3 install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118
#   - Install and set up ninja for faster compilation of PyTorch3D.
RUN pip install ninja~=1.11
ENV MAX_JOBS=$(nproc)
RUN pip install "wheel~=0.43"
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@eaf0709d6af0025fe94d1ee7cec454bc3054826a"
RUN pip install -r ${NEUSURFEMB_ROOT}/third_party/NeuS2/requirements.txt
# - Install mmtracking and download the necessary checkpoints.
WORKDIR ${NEUSURFEMB_ROOT}/third_party/mmtracking
RUN git checkout e79491ec8f0b8c86fda947fbaaa824c66ab2a991 && git submodule update --init --recursive
ENV MMCV_WITH_OPS=1
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
RUN pip install mmdet==2.28.2
RUN pip install -r requirements/build.txt
RUN pip install -v -e .
RUN mkdir checkpoints
WORKDIR ${NEUSURFEMB_ROOT}/third_party/mmtracking/checkpoints
RUN wget -c https://download.openmmlab.com/mmtracking/sot/mixformer/mixformer_cvt_500e_got10k/mixformer_cvt_500e_got10k.pth
# - Install the dependencies of Hierarchical-Localization.
WORKDIR ${NEUSURFEMB_ROOT}/third_party/Hierarchical-Localization
RUN git checkout a9ee933e4ed9e82709f2fdbae0b9d1013273c8b4 && git submodule update --init --recursive
RUN pip install -e .
# - Install the dependencies of surfemb.
WORKDIR ${NEUSURFEMB_ROOT}/third_party/surfemb
RUN git checkout 9e65db9cb7c83b41c012ad853585e7a86a113fef && git submodule update --init --recursive
RUN pip install -r requirements.txt
#   - Install torch_scatter.
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1%2Bcu118.html
# - Download the SurfEmb data necessary for evaluation.
WORKDIR ${NEUSURFEMB_ROOT}/third_party/surfemb/
RUN apt update && apt install -y unzip
RUN wget https://github.com/rasmushaugaard/surfemb/releases/download/v0.0.1/inference_data.zip && \
    unzip -q inference_data.zip && \
    rm inference_data.zip
# - Install segment-anything.
WORKDIR ${NEUSURFEMB_ROOT}/third_party/segment-anything
RUN git checkout 6fdee8f2727f4506cfbbe553e23b895e27956588 && git submodule update --init --recursive
RUN pip install -e .
RUN pip install opencv-python~=4.8.1.78 onnxruntime-gpu==1.15.0 onnx==1.14.1
#   - Download a model checkpoint.
RUN mkdir ${NEUSURFEMB_ROOT}/third_party/segment-anything/weights
WORKDIR ${NEUSURFEMB_ROOT}/third_party/segment-anything/weights
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
#   - Convert the model checkpoint to ONNX and quantize it.
RUN python ${NEUSURFEMB_ROOT}/third_party/segment-anything/scripts/export_onnx_model.py \
    --return-single-mask \
    --checkpoint ${NEUSURFEMB_ROOT}/third_party/segment-anything/weights/sam_vit_h_4b8939.pth \
    --output ${NEUSURFEMB_ROOT}/third_party/segment-anything/weights/sam_onnx_multihyp.onnx \
    --model-type vit_h \
    --quantize-out ${NEUSURFEMB_ROOT}/third_party/segment-anything/weights/sam_onnx_quantized.onnx
# - Install YOLOv8.
RUN pip install ultralytics~=8.0.120
# - Install BOP Toolkit and its dependencies.
WORKDIR ${NEUSURFEMB_ROOT}/third_party/bop_toolkit
RUN git checkout 7c79c7eef0dbbd000dc2b1bf5b48450385afd366 && git submodule update --init --recursive
RUN pip install Cython==0.29.24
RUN pip install -r requirements.txt -e .
# - Install other dependencies:
WORKDIR ${NEUSURFEMB_ROOT}
RUN pip install -r requirements.txt
# - Install CGAL [only needed for real data].
#   - Install dependencies.
RUN apt update && apt install -y libgmp-dev libmpfr-dev
WORKDIR ${NEUSURFEMB_ROOT}/third_party/cgal
RUN git checkout 157782a45f047d3f261ed3b40785e0cdc3dff14b && git submodule update --init --recursive
RUN mkdir build
WORKDIR build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release
RUN make install
# - Compile the bbox_estimator [only needed for real data].
WORKDIR ${NEUSURFEMB_ROOT}/bbox_estimator
RUN mkdir build 
WORKDIR ${NEUSURFEMB_ROOT}/bbox_estimator/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release
RUN make -j12
# - Install the package.
WORKDIR ${NEUSURFEMB_ROOT}
RUN pip install -e .
# - Set the correct version of setuptools (otherwise `mmcv` cannot be imported).
RUN pip install setuptools==69.5.1
RUN apt update && apt install -y ffmpeg
# Install additional required `apt` dependencies:
RUN apt update && apt-get install -y python3-tk libgtk-3-dev python-wxtools
# Install ROS-related dependencies in the virtualenv:
RUN pip install rospkg~=1.5.1 pycryptodomex~=3.20.0 gnupg~=2.3.1 igraph~=0.11.6
RUN pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-20.04/wxPython-4.2.0-cp38-cp38-linux_x86_64.whl
# Remove the output `models/` in SurfEmb, since it will be symlinked when training (cf. Step 8 `neusurfemb/dataset_scripts/example_pipeline_run.sh`).
RUN rm -r ${NEUSURFEMB_ROOT}/third_party/surfemb/data/models;