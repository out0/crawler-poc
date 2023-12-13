#! /bin/bash
CARLA_SIMULATION_IMAGE=carlasim-dev-img

if [[ "$(docker images -q $CARLA_SIMULATION_IMAGE 2>/dev/null)" == "" ]]; then
    docker build -t $CARLA_SIMULATION_IMAGE -f ../dev-env/containers/carlasim/img/Dockerfile ../dev-env/containers/carlasim
 fi
  
nvidia-docker run -it \
    --rm --privileged \
    --gpus device=0 \
    --detach \
    -e NVIDIA_VISIBLE_DEVICES=1 \
    -e CUDA_VISIBLE_DEVICES=1 \
    -e SDL_HINT_CUDA_DEVICE=1 \
    --net host \
    --user carla \
    --env DISPLAY=$DISPLAY \
     --env QT_X11_NO_MITSHM=1 \
    --env PULSE_SERVER=unix:/run/user/1000/pulse/native \
     --volume /tmp/.X11-unix:/tmp/.X11-unix \
     --runtime nvidia \
     --expose 2000-21000 \
     -v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d \
     $CARLA_SIMULATION_IMAGE \
    /bin/bash -c /home/carla/CarlaUE4.sh

docker run \
    --rm --privileged \
    --gpus device=0 \
    --detach \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e SDL_HINT_CUDA_DEVICE=0 \
    --net host \
    --user out0 \
    --env DISPLAY=$DISPLAY \
    --env QT_X11_NO_MITSHM=1 \
    --env PULSE_SERVER=unix:/run/user/1000/pulse/native \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume /home/cristiano/Documents/Projects/Mestrado/Project/crawler-poc/notebooks:/home/out0/notebooks \
    --volume /home/cristiano/Documents/Projects/Mestrado/Project/crawler-poc/results:/home/out0/results \
    --runtime nvidia \
    --expose 2000-21000 \
     driveless-dev-cuda-img

#sleep 20

#gst-launch-1.0 -v udpsrc port=20000 caps = \"application/x-rtp, media=\(string\)video, clock-rate=\(int\)90000, encoding-name=\(string\)H264, payload=\(int\)96\" ! rtph264depay ! decodebin ! videoconvert ! autovideosink 
#gst-launch-1.0 -v udpsrc port=20001 caps = \"application/x-rtp, media=\(string\)video, clock-rate=\(int\)90000, encoding-name=\(string\)H264, payload=\(int\)96\" ! rtph264depay ! decodebin ! videoconvert ! autovideosink
#gst-launch-1.0 -v udpsrc port=20002 caps = \"application/x-rtp, media=\(string\)video, clock-rate=\(int\)90000, encoding-name=\(string\)H264, payload=\(int\)96\" ! rtph264depay ! decodebin ! videoconvert ! autovideosink


#gst-launch-1.0 -v udpsrc port=20003 caps = \"application/x-rtp, media=\(string\)video, clock-rate=\(int\)90000, encoding-name=\(string\)H264, payload=\(int\)96\" ! rtph264depay ! decodebin ! videoconvert ! autovideosink
