# Deploy SAX model to Vertex Prediction

SAX has implementation for Gemma, Llama2 and GPT-J models that can be deployed
on GPU or TPU.

## Get model checkpoint

For the purpose of this sample we assume that model is deployed to
\${PROJECT_ID} GCP project and model is saved to
gs://$(BUCKET_NAME)/\<model_folder\>/checkpoint_00000000 GCS folder.


### Gemma
To get access to the Gemma models, you must sign in to the [Kaggle](https://www.kaggle.com/) platform,
sign the license consent agreement, and get a Kaggle API token as described in
https://github.com/Kaggle/kaggle-api/blob/main/README.md#kaggle-api.

You can download `google/gemma/pax/2b-it/1` and `google/gemma/pax/7b-it/1` models.

```sh
pip install kaggle --break-system-packages
mkdir -p /data/gemma_2b-it &&\
kaggle models instances versions download google/gemma/pax/2b-it/1 --untar -p /data/gemma_2b-it
gsutil -m cp -R /data/gemma_2b-it/* gs://${BUCKET_NAME}/gemma_2b-it/
```

### LLama2

For LLama2, you first need to download model weights from Meta
https://ai.meta.com/llama/. Then convert weights to SAX format using
convert_llama_ckpt.py script from tools folder as described in
[here](../../README.md?tab=readme-ov-file#use-sax-to-load-llama-7b13b70b-model).

### GPT-J

For GPT-J follow instructions in [convert_gptj_ckpt.py](../../saxml/tools/convert_gptj_ckpt.py) script.

## Obtain prebuilt SAX containers

Prebuilt containers are available at `us-docker.pkg.dev/vertex-ai/prediction/sax-gpu:latest` and
`us-docker.pkg.dev/vertex-ai/prediction/sax-tpu:latest`.

You can use those, but you first need to copy them into your project:

```bash
# For GPU
docker tag us-docker.pkg.dev/vertex-ai/prediction/sax-gpu:latest us-docker.pkg.dev/${PROJECT_ID}/prediction/sax-vertex-cuda:latest
docker push us-docker.pkg.dev/${PROJECT_ID}/prediction/sax-vertex-cuda:latest

# For TPU
docker tag us-docker.pkg.dev/vertex-ai/prediction/sax-tpu:latest us-docker.pkg.dev/${PROJECT_ID}/prediction/sax-vertex-tpu:latest
docker push us-docker.pkg.dev/${PROJECT_ID}/prediction/sax-vertex-tpu:latest
```

## Build Vertex TPU or GPU container

If you need to do any changes to the models you can build containers yourself.
Follow steps in [docker README file](../tools/docker/README.md) to build either
`sax-vertex-cuda` or `sax-vertex-tpu` container.

## (Optional) Test container locally

If you are running steps on GPU or TPU VM you can run container locally to test
that it works. Or you can run a small model on CPU. For CPU models either
`sax-vertex-cuda` or `sax-vertex-tpu` container would work.

```bash
docker run \
-e JAX_PLATFORMS='cpu' \
-e SAX_ROOT=/tmp/sax-test-root/ \
-p 8500:8500 -p 8888:8888 -p 10000:10000 -p 14005:14005 \
sax-vertex-cuda:latest \
  --port=8888 \
  --grpc_port=8500 \
  --model_path=saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd2BTest \
  --jax_profiler_port=14005 \
  --platform_chip=cpu \
  --platform_topology=1
```

For running GPU model:

```bash
docker run --gpus=all --shm-size=2g \
-e JAX_PLATFORMS='cuda' \
-e SAX_ROOT=/tmp/sax-test-root/ \
-p 8500:8500 -p 8888:8888 -p 10000:10000 -p 14005:14005 \
sax-vertex-cuda:latest \
  --port=8888 \
  --grpc_port=8500 \
  --model_path=saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd2BTest \
  --jax_profiler_port=14005 \
  --platform_chip=l4 \
  --platform_topology=1
```

Once model is running you can check that it is up and running:

```bash
 curl http://localhost:8888/health
```

Send a request to HTTP server:

```python
import json
import requests

payload={"instances": [{"text_batch": "Q:1+1=2 A:2+2="}]}

response = requests.post(f'http://localhost:8888/predict', data=json.dumps(payload), timeout=180)
response.text
 ```

### (Optional) Profiling model running locally

You can profile models running locally using TensorFlow profiler.
Make sure that you have latest TensorFlow version installed.

First make sure that your model is up and running and is responding to requests.

Issue a profile request:

```python
import tensorflow as tf

tf.profiler.experimental.client.trace(
    'grpc://localhost:14005',
    logdir='gs://${PROJECT_ID}/profiles',
    duration_ms=5000,
    num_tracing_attempts=1,
    options=tf.profiler.experimental.ProfilerOptions(
        host_tracer_level=2,    # should be 0 for TPU profiles
        python_tracer_level=1,  # necessary for JAX profiling
        device_tracer_level=1,
        delay_ms=None)
    )
```

Wait a moment until you see that profiling has started in SAX logs, send a
request to model you want to profile.

Once profile is collected, you can inspect it using `tensorboard`, note that you
might need to install profiler plugin separately.

```bash
# Install tensorboard
pip install tensorboard

# Install profiler plugin
pip install -U tensorboard_plugin_profile

# Start tensorboard
tensorboard -logdir=gs://${PROJECT_ID}/profiles
```

## Publish container to Artifact Repository

In order to deploy model to Vertex Prediction, you need to upload your SAX
container to Artifact Registry first. If you haven't used it before, you might
need to configure it. Run following commands once to configure Artifact
Registry:

```bash
gcloud auth configure-docker us-docker.pkg.dev --quiet
gcloud artifacts repositories create prediction --location=us --repository-format=docker
```

And then upload container to Artifact Registry.

```bash
# For GPU
docker tag sax-vertex-cuda:latest us-docker.pkg.dev/${PROJECT_ID}/prediction/sax-vertex-cuda:latest
docker push us-docker.pkg.dev/${PROJECT_ID}/prediction/sax-vertex-cuda:latest

# For TPU
docker tag sax-vertex-tpu:latest us-docker.pkg.dev/${PROJECT_ID}/prediction/sax-vertex-tpu:latest
docker push us-docker.pkg.dev/${PROJECT_ID}/prediction/sax-vertex-tpu:latest
```

## Deploy SAX container to Vertex Prediction

If you haven't already, please install Vertex AI SDK:
https://cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-python-sdk

### Initialize SDK

```python
from google.cloud import aiplatform

aiplatform.init(project='${PROJECT_ID}', location='us-west1')
```

### Create endpoint

```python
endpoint = aiplatform.Endpoint.create(display_name='SAX endpoint')
```

### Upload model

To upload model for L4 GPUs, you can use following configuration:

```python
model = aiplatform.Model.upload(
    display_name='SAX Gemma 2B model for g2-standard-48',
    artifact_uri="gs://${PROJECT_ID}//gemma_2b-it/",
    serving_container_image_uri="us-docker.pkg.dev/${PROJECT_ID}/prediction/sax-vertex-cuda:latest",
    serving_container_args=[
      "--model_path=saxml.server.pax.lm.params.gemma.Gemma2BFP16",
      "--platform_chip=l4",
      "--platform_topology=4",
      "--ckpt_path_suffix=checkpoint_00000000",
      "--bypass_health_check=true",
      "--prediction_timeout_seconds=600",
    ],
     serving_container_ports=[8502]
)
```

To upload model for v5e TPUs refer to Vertex AI Prediction CloudTPU deployment
guide: https://cloud.google.com/vertex-ai/docs/predictions/use-tpu

```python
model = aiplatform.Model.upload(
    display_name='SAX Gemma 2B for ct5lp-hightpu-4t',
    artifact_uri="gs://${PROJECT_ID}/gemma_2b-it/",
    serving_container_image_uri="us-docker.pkg.dev/${PROJECT_ID}/prediction/sax-vertex-tpu:latest",
    serving_container_args=[
      "--model_path=saxml.server.pax.lm.params.gemma.Gemma2BFP16",
      "--platform_chip=tpuv5e",
      "--platform_topology=2x2"
      "--ckpt_path_suffix=checkpoint_00000000",
      "--bypass_health_check=true",
      "--prediction_timeout_seconds=600",
    ],
     serving_container_ports=[8502]
)
```

### Deploy model

To deploy on 4xL4 GPUs:

```python
endpoint_deployed = model.deploy(
    endpoint=endpoint,
    deployed_model_display_name='SAX Gemma 2B on 4xL4 deployed model',
    machine_type='g2-standard-48',
    accelerator_type="NVIDIA_L4",
    accelerator_count=4,
    traffic_percentage=100,
    sync=True
)
```

To deploy on tpuv5e:

```python
endpoint_deployed = model.deploy(
    endpoint=endpoint,
    deployed_model_display_name='SAX Gemma 2B on ct5lp-hightpu-4t deployed model',
    machine_type='ct5lp-hightpu-4t',
    traffic_percentage=100,
    sync=True
)
```

### Send prediction request

```python
endpoint_deployed.predict(
    instances=[{"text_batch": "The capital of US is"}]
)
```
