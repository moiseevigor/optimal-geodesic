# Optimal Geodesic
Search for Optimal Geodesic on Manifolds such as R^2 \times S^1, R^3 \times S^2.

## Theory

Geometry library https://github.com/geomstats/geomstats/tree/master/geomstats/geometry

### Circle 

Implicit surface formula in (x,y,z) \in R^3

```
x^2 + y^2 + z^2 = 1
```

Geodeics parametrized by time

```
\dot x = cos(th)*sin(th)
\dot y = sin(th)
\dot z = ??
```

### Ellipse
Implicit surface formula in (x,y,z) \in R^3

```
x^2/a + y^2/b + z^2 = 1
```

Geodeics parametrized by time

```
\dot x = a*cos(th)*sin(th)
\dot y = b*sin(th)
\dot z = ??
```

### Torus

## Heisenberg group, H(2)??

### E(2) = R^2 \times S^1

### E(3) = R^3 \times S^2

SE(3) https://geomstats.github.io/notebooks/04_practical_methods__from_vector_spaces_to_manifolds.html#Geodesics-on-the-special-euclidean-group-SE(3)

### E(8) - standard model group

### Build & Run

```
docker build -t pytorch -f Dockerfile .
```

Run in interactive mode with X11 on local machine

Set permissions to connect to X11

```
xhost +
```

run docker with GUI on Mac OS

```
docker run --rm --gpus all -it --env="DISPLAY" -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" -v $PWD:/app pytorch python /app/src/r2s1/r2s1.py
```

on Mac OS

```
docker run --rm -it -e DISPLAY=host.docker.internal:0 -v $PWD:/app pytorch python /app/src/r2s1/r2s1.py
```

