# SAX Docker Files (OSS Only)
* `Dockerfile.dev`: A build image to build SAX binaries, including `saxutil`, `admin config`, `admin server` and `model server`. It provides the binary for the following corresponding runtime image which means this image needs to build first.

* `Dockerfile.admin`: SAX admin server runtime image (with extra `saxutil` installed).

* `Dockerfile.model`: SAX model server runtime image.

* `Dockerfile.util`: saxutil cli runtime image.

## Run quick build and test on GCE CPU VM
Set up docker permission:

```
sudo usermod -a -G docker ${USER}
newgrp docker
```

Build all images (assume you are in this directory):

NOTE: For building CUDA image, copy requirements.txt file as described in
https://github.com/google/saxml/tree/main?tab=readme-ov-file#start-the-sax-gpu-model-server

```
SAX_ROOT_PATH=$(git rev-parse --show-toplevel)
docker build -f Dockerfile.dev ${SAX_ROOT_PATH} -t sax-dev
docker build -f Dockerfile.admin . -t sax-admin
docker build -f Dockerfile.model . -t sax-model
docker build -f Dockerfile.util . -t sax-util
docker build --build-arg JAX_PLATFORMS=tpu -f Dockerfile.vertex . -t sax-vertex-tpu
docker build --build-arg JAX_PLATFORMS=cuda -f Dockerfile.vertex . -t sax-vertex-cuda
```

Create a Cloud Storage bucket:

```
GSBUCKET=${USER}-sax-data
gcloud storage buckets create gs://${GSBUCKET}
```

Run admin server (in background):

```
docker run -d -e GSBUCKET=${GSBUCKET} sax-admin
```

To check the admin server log:

```
CONTAINER_ID=$(docker ps | grep "sax-admin" | awk '{print $1}') && docker logs ${CONTAINER_ID}
```


Run model server (in background):

```
docker run -d \
  -e JAX_PLATFORMS=cpu \
  -e SAX_ROOT=gs://${GSBUCKET}/sax-root \
  sax-model \
  --sax_cell=/sax/test \
  --port=10001 \
  --platform_chip=cpu \
  --platform_topology=1
```

To check the model server log (take around 1 min to be ready):

```
CONTAINER_ID=$(docker ps | grep "sax-model" | awk '{print $1}') && docker logs ${CONTAINER_ID}
```

Publish model:

```
docker run sax-util --sax_root=gs://${GSBUCKET}/sax-root \
  publish \
  /sax/test/lm2b \
  saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd2BTest \
  None \
  1
```

When the model is loaded, issue a query:

```
docker run sax-util --sax_root=gs://${GSBUCKET}/sax-root \
  lm.generate \
  /sax/test/lm2b \
  "Q: Who is Harry Porter's mother? A: "
```
