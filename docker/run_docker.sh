#!/bin/bash

# Based on https://github.com/ethz-asl/moma/blob/master/docker/run_docker.sh
# and https://github.com/ethz-asl/wavemap/blob/main/tooling/packages/wavemap_utils/scripts/run_in_docker.sh.

# Default options
DOCKER=neusurfemb
DOCKERFILE=main.Dockerfile
NAME=neusurfemb
BUILD=false

help()
{
    echo "Usage: run_docker.sh [ -b | --build <dockerfile name> ]
               [ -n | --name <docker name> ] [ -d | --data-dir <host data dir> ]
               [ -h | --help  ]"
    exit 2
}

SHORT=b:,n:,d:,h
LONG=build:,name:,data-dir:,help
OPTS=$(getopt -a -n run_docker --options $SHORT --longoptions $LONG -- "$@")
echo $OPTS

eval set -- "$OPTS"

while :
do
  case "$1" in
    -b | --build )
      BUILD="true"
      DOCKERFILE="$2"
      shift 2
      ;;
    -n | --name )
      NAME="$2"
      shift 2
      ;;
    -d | --data-dir )
      DATA_DIR="$2"
      shift 2
      ;;
    -h | --help)
      help
      ;;
    --)
      shift;
      break
      ;;
    *)
      echo "Unexpected option: $1"
      help
      ;;
  esac
done

if [[ -z $@ ]]
then
  COMMAND="bash";
else
  COMMAND="$@";
fi

if [[ -z ${DATA_DIR} ]]
then
  DATA_DIR_STR=""
else
  DATA_DIR_STR="-v ${DATA_DIR}:/home/data:rw"
fi

if [ "$BUILD" = true ]; then
     # BuildKit currently needs to be disabled to allow using NVIDIA Runtime
     # during build (cf., e.g.,
     # https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime/75629058#75629058).
     DOCKER_BUILDKIT=0 docker build -f $DOCKERFILE -t $DOCKER .
fi

xhost + local:docker

# Check if there are multiple commands.
# - Hack: Write the multiple commands to a temporary bash script and execute the
#   script.
tmp=${COMMAND//&&/$'\2'}
IFS=$'\2' read -a arr <<< "$tmp"
NUMBER_OF_COMMANDS=${#arr[@]};
if [ ${NUMBER_OF_COMMANDS} -gt 1 ]; then
  TMPFILE=$(tempfile -d ${DATA_DIR});
  mv $TMPFILE $TMPFILE.sh;
  TMPFILE=$TMPFILE.sh;
  echo -e '#!/bin/bash\n' > $TMPFILE;
  echo -e "${COMMAND}" >> $TMPFILE;
  COMMAND="/bin/bash /home/data/$(basename ${TMPFILE})";
fi

docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    ${DATA_DIR_STR} \
    --runtime=nvidia --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,display \
    --net=host \
    --name=$NAME \
    --shm-size=16gb \
    ${DOCKER} \
    ${COMMAND}

# Remove the temporary file containing the multiple commands, if needed.
if [ ${NUMBER_OF_COMMANDS} -gt 1 ]; then
  rm ${TMPFILE};
fi

echo "Done."