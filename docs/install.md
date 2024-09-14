# Installation
The following installation instructions were tested on a Ubuntu 20.04 machine with a NVIDIA GeForce RTX 3080 Laptop GPU and NVIDIA Driver Version 535. We assume CUDA Toolkit >= 11.8 to be already installed on your computer. 

## Docker installation
We recommend installing our pipeline using the [provided Dockerfile](../docker/main.Dockerfile).
If you do not have Docker installed on your computer, please follow the [official instructions](https://docs.docker.com/engine/install/), including the [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/).
Additionally, install the NVIDIA Container Toolkit, required to access and use the GPU within the container, following [these instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Additionally, make sure that the line starting by `no-cgroups` in your `/etc/nvidia-container-runtime/config.toml` file is either commented or set to `no-cgroups = false`. Finally, add the line `"default-runtime": "nvidia",` to the `/etc/docker/daemon.json` file, so that the content of the file looks as follows:
```json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
```
and run `systemctl restart docker` afterwards.

You can then build the Docker image by navigating to the [`docker/` folder](../docker/) of this repo and running:
```bash
./run_docker.sh -b main.Dockerfile
```

You can expect the build process to take between 30 and 50 minutes.

## Manual installation
Alternatively, follow the steps below to manually install the pipeline on your system without using Docker.

### Python module installation
In the following, the root folder is assumed to be called `${NEUSURFEMB_ROOT}`.
- Clone the repo and its submodules:
    ```bash
    git clone https://github.com/ethz-asl/neusurfemb.git;
    cd neusurfemb;
    git submodule update --init --recursive;
    export NEUSURFEMB_ROOT=$PWD;
    ```
- Create a virtualenv:
    ```bash
    export NEUSURFEMB_VIRTUALENV=~/.virtualenvs/neusurfemb;
    mkdir -p ${NEUSURFEMB_VIRTUALENV};
    python3 -m venv ${NEUSURFEMB_VIRTUALENV};
    source ${NEUSURFEMB_VIRTUALENV}/bin/activate;
    pip install --upgrade pip;
    ```
- Install NeuS2:
    ```bash
    cd ${NEUSURFEMB_ROOT}/third_party/NeuS2;
    git checkout 01f322a6701a762564e3bc250ac561bdd7b7659e && git submodule update --init --recursive;
    cmake . -B build;
    cmake --build build --config RelWithDebInfo -j;
    cp ${NEUSURFEMB_ROOT}/third_party/NeuS2/build/pyngp.cpython-38-x86_64-linux-gnu.so ${NEUSURFEMB_VIRTUALENV}/lib/python3.8/site-packages/;
    ```
    If the above steps failed with the error `Failed to detect a default CUDA architecture.`, make sure that the CUDA bins are in your path (`export PATH=/usr/local/cuda/bin:$PATH`).
- Install dependencies of NeuS (here assuming CUDA Toolkit 11.8 is installed):
    ```bash
    # Adjust this based on your CUDA toolkit version (which can be looked up with `nvcc --version`), following
    # https://pytorch.org/get-started/previous-versions/#v201.
    pip3 install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118;
    # Install and set up ninja for faster compilation of PyTorch3D.
    pip install ninja~=1.11;
    export MAX_JOBS=$(nproc);
    pip install "wheel~=0.43";
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@eaf0709d6af0025fe94d1ee7cec454bc3054826a";
    pip install -r ${NEUSURFEMB_ROOT}/third_party/NeuS2/requirements.txt;
    ```
- Install `mmtracking` and download the necessary checkpoints:
    ```bash
    pushd ${NEUSURFEMB_ROOT}/third_party/mmtracking;
    git checkout e79491ec8f0b8c86fda947fbaaa824c66ab2a991 && git submodule update --init --recursive;
    export MMCV_WITH_OPS=1;
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html;
    pip install mmdet==2.28.2;
    pip install -r requirements/build.txt;
    pip install -v -e .;
    mkdir checkpoints;
    cd checkpoints;
    wget -c https://download.openmmlab.com/mmtracking/sot/mixformer/mixformer_cvt_500e_got10k/mixformer_cvt_500e_got10k.pth;
    ```
- Install the dependencies of `Hierarchical-Localization`:
    ```bash
    cd ${NEUSURFEMB_ROOT}/third_party/Hierarchical-Localization;
    git checkout a9ee933e4ed9e82709f2fdbae0b9d1013273c8b4 && git submodule update --init --recursive;
    pip install -e .;
    ```
- Install the dependencies of `surfemb`:
    ```bash
    cd ${NEUSURFEMB_ROOT}/third_party/surfemb;
    git checkout 9e65db9cb7c83b41c012ad853585e7a86a113fef && git submodule update --init --recursive;
    pip install -r requirements.txt;
    # Install torch_scatter.
    export MAX_JOBS=$(nproc);
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1%2Bcu117.html
    ```
- Download the SurfEmb data necessary for evaluation:
    ```bash
    cd ${NEUSURFEMB_ROOT}/third_party/surfemb/;
    wget https://github.com/rasmushaugaard/surfemb/releases/download/v0.0.1/inference_data.zip;
    unzip -q inference_data.zip && rm inference_data.zip;
    ```
- Install `segment-anything`:
    ```bash
    cd ${NEUSURFEMB_ROOT}/third_party/segment-anything;
    git checkout 6fdee8f2727f4506cfbbe553e23b895e27956588 && git submodule update --init --recursive;
    pip install -e .;
    pip install opencv-python~=4.8.1.78 onnxruntime-gpu==1.15.0 onnx==1.14.1;
    # Download a model checkpoint.
    mkdir ${NEUSURFEMB_ROOT}/third_party/segment-anything/weights;
    cd ${NEUSURFEMB_ROOT}/third_party/segment-anything/weights;
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth;    
    # Convert the model checkpoint to ONNX and quantize it.
    python ${NEUSURFEMB_ROOT}/third_party/segment-anything/scripts/export_onnx_model.py --return-single-mask --checkpoint ${NEUSURFEMB_ROOT}/third_party/segment-anything/weights/sam_vit_h_4b8939.pth --output ${NEUSURFEMB_ROOT}/third_party/segment-anything/weights/sam_onnx_multihyp.onnx --model-type vit_h --quantize-out  ${NEUSURFEMB_ROOT}/third_party/segment-anything/weights/sam_onnx_quantized.onnx;
    ```
- Install `YOLOv8`:
    ```bash
    pip install ultralytics~=8.0.120;
    ```
- Install BOP Toolkit and its dependencies:
    ```bash
    cd ${NEUSURFEMB_ROOT}/third_party/bop_toolkit;
    git checkout 7c79c7eef0dbbd000dc2b1bf5b48450385afd366 && git submodule update --init --recursive;
    pip install Cython==0.29.24;
    pip install -r requirements.txt -e .;
    ```
- Install other dependencies:
    ```bash
    cd ${NEUSURFEMB_ROOT};
    pip install -r requirements.txt;
    ```
- Install CGAL [only needed for real data]:
    ```bash
    # Install dependencies.
    sudo apt install libgmp-dev libmpfr-dev;
    cd ${NEUSURFEMB_ROOT}/third_party/cgal;
    git checkout 157782a45f047d3f261ed3b40785e0cdc3dff14b && git submodule update --init --recursive;
    mkdir build && cd build;
    cmake .. -DCMAKE_BUILD_TYPE=Release;
    sudo make install;
    ```
- Compile the [`bbox_estimator`](../bbox_estimator) [only needed for real data]:
    ```bash
    cd ${NEUSURFEMB_ROOT}/bbox_estimator && mkdir build && cd build;
    cmake .. -DCMAKE_BUILD_TYPE=Release;
    make -j12;
    ```
- Install the package:
    ```bash
    cd ${NEUSURFEMB_ROOT};
    pip install -e .;
    ```
- Set the correct version of setuptools (otherwise `mmcv` cannot be imported):
    ```bash
    pip install setuptools==69.5.1;
    ```

### ROS modules installation [Optional]
Some modules (camera calibration, image undistortion) in our repo depend on Robot Operating System (ROS) packages. To use them, please install ROS Noetic using the official [installation instructions](http://wiki.ros.org/noetic/Installation/Ubuntu) and additionally install the following dependencies:
```bash
sudo apt update
sudo apt install git build-essential
sudo apt install python3-catkin-tools python3-vcstool python3-rosdep
```

A catkin workspace with the required packages as submodule is already provided in the [`ros/catkin_ws`](../ros/catkin_ws/) folder, which we will now refer to as `${CATKIN_WS}`.
- Check out the correct version of each package:
    ```bash
    cd ${CATKIN_WS};
    catkin init && catkin config --extend /opt/ros/noetic --cmake-args -DCMAKE_BUILD_TYPE=Release;
    cd ${CATKIN_WS}/src/catkin_simple;
    git checkout 0e62848b12da76c8cc58a1add42b4f894d1ac21e && git submodule update --init --recursive;
    cd ${CATKIN_WS}/src/kalibr;
    git checkout 3c2856c77bc1ec22a5d02d49a0e889df0356cb17 && git submodule update --init --recursive;
    cd ${CATKIN_WS}/src/image_undistort;
    git checkout a8fa9cdcf2c25f8ca83556b7bfb891fe64a3612b && git submodule update --init --recursive;
    ```
- Install the dependencies and build the catkin workspace:
    ```bash
    sudo apt update && sudo apt install -y libv4l-dev libsuitesparse-dev;
    cd ${CATKIN_WS}/src;
    catkin build -j12;
    source ${CATKIN_WS}/devel/setup.bash;
    ```
- Install additional `pip` dependencies in the virtualenv:
    ```bash
    source ${NEUSURFEMB_VIRTUALENV}/bin/activate;
    pip install pyx~=0.16 empy~=4.1;
    ```
- Install additional required `apt` dependencies:
    ```bash
    apt update && apt-get install -y python3-tk libgtk-3-dev;
    ```
- Install ROS-related dependencies in the virtualenv:
    ```bash
    source ${NEUSURFEMB_VIRTUALENV}/bin/activate;
    pip install rospkg~=1.5.1 pycryptodomex~=3.20.0 gnupg~=2.3.1 igraph~=0.11.6;
    pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-20.04/wxPython-4.2.0-cp38-cp38-linux_x86_64.whl;
    ```
