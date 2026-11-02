#!/bin/bash
docker run --gpus all -it --rm --shm-size=8g \
    -v $(pwd):/leap \
    -v /data:/data \
    leap:latest
