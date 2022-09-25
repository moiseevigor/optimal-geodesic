# Optimal Geodesic
Search for Optimal Geodesic on Manifolds such as R^2 \times S^1, R^3 \times S^2.

## Theory

## R^2 \times S^1

## R^3 \times S^2.

## Installation 

## Build & Run

Build

```
docker build -t pytorch-cpu -f Dockerfile.cpu .
```

Run in interactive mode with X11 on local machine

Set permissions to connect to X11

```
xhost +
```

run docker with GUI

```
docker run --rm -it -e DISPLAY=host.docker.internal:0 -v $PWD:/app pytorch-cpu python /app/src/r2s1/opt_geod.py
```