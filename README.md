# POC: Crawler Vision Module

Put trained models in `assets/`.

Download onnx trained model on bosque data [here](https://drive.google.com/file/d/12A1e7hoqCTDovk6MvJoOUSHVZAy_64Am/view?usp=sharing).

Download bosque data [here](https://drive.google.com/file/d/14kEL1jbS1iEOQ9scYj-UrGlPnyhomD-m/view?usp=sharing).

Run:
```
xhost +

docker run --rm -v $HOME:$HOME -e DISPLAY=$DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix   -it tldrafael/ubuntu20-py310-cu117:latest bash

cd $HERE
python3 vision.py model.onnx dataset.mp4
```

# Objetivos

1. Organizar o notebook para catpura, segmentação, BEV, planning, output de imagem (1 frame)  (PC)

2. Organizar o notebook para captura, segmentação, BEV, planning, output de vídeo  (PC)

3. Organizar o notebook para captura, segmentação, BEV, planning, output de vídeo  (Jetson)

