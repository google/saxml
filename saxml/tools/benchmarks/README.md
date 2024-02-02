# OSS SAXML Benchmarking
## Run the Benchmarking on Cloud TPU

Note: SAXML respects and checks the EOS and the max decode steps after each decode step even for test mode with random weight initialization. This makes the benchmarking result as the same as the real serving/inference scenario (which does not know the output length ahead of time).

Assume you are using the SAX docker image for SAX deployment and already log
into the Cloud TPU VM.

### Step1. Set up Environment Variable

```
# Image Paths.
SAX_UTIL_IMAGE=YOUR_SAX_UTIL_IMAGE
SAX_ADMIN_IMAGE=YOUR_SAX_ADMIN_IMAGE
SAX_MODEL_IMAGE=YOU_MODEL_IMAGE

# Your GCS bucket to store the SAX config.
GSBUCKET=YOUR_GCSBUCKET

# Set up chip type.
PLATFORM_CHIP=tpuv5e

# SAX Model Class, e.g., saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd2BTest.
SAX_CLASS=YOUR_SAX_CLASS
# Model name in the SAX cluster.
MODEL_NAME=YOUR_MODEL_NAME
# Checkpoint Path.
# If set the CKPT_PATH as "None", the SAX will initilize the weight with random
# number.
CKPT_PATH=YOUR_CKPT_PATH
```

### Step2. Deploy SAX on the Cloud.
```
# Launch Admin Server.
docker run -d --network host -e GSBUCKET=${GSBUCKET} ${SAX_ADMIN_IMAGE}

# Launch Model Server.
# For multi-host, you need to run this on every Cloud TPU host.
docker run -d --privileged --network host \
  -e SAX_ROOT=gs://${GSBUCKET}/sax-root \
  ${SAX_MODEL_IMAGE} \
  --sax_cell=/sax/test \
  --port=10001 \
  --platform_chip=${PLATFORM_CHIP}
```

### Step3. Publish Model
Please change the model serving config based on the your setup.

```
alias saxutil='docker run ${SAX_UTIL_IMAGE} --sax_root=gs://${GSBUCKET}/sax-root'

saxutil publish /sax/test/${MODEL_NAME} \
  ${SAX_CLASS} \
  ${CKPT_PATH} \
  1 \
  BATCH_SIZE=32 \
  NUM_SAMPLES=1 \
  BUCKET_KEYS=[256,512,1024] \
  TOP_K=40 \
  INPUT_SEQ_LEN=1024 \
  MAX_DECODE_STEPS=1024 \
  BATCH_WAIT_SECS=1 \
  EXTRA_INPUTS={\"temperature\":1\,\"per_example_max_decode_steps\":1024\,\"per_example_top_k\":40\,\"per_example_top_p\":0.9}
```

### Step4. Run benchmarking
If you don't specify the dataset, the benchmarking script will generate the dummy prompts with length as `--max-input-length` (default 1024).


```
# Clone the SAX repo.
git clone https://github.com/google/saxml.git
cd saxml/saxml/tools/benchmarks

# Download the SAX client.
gsutil cp -r gs://cloud-tpu-inference-public/benchmark/sax_client .
export PYTHONPATH=${PYTHONPATH}:${PWD}/sax_client

# Install deps:
pip install -r requirements.txt

# Set SAX_ROOT variable.
export SAX_ROOT=gs://${GSBUCKET}/sax-root
```

```
# Run benchmarking.
python3 benchmark_serving.py \
  --model=/sax/test/${MODEL_NAME} \
  --dataset=${LOCAL_DATASET_PATH} \
  --num-prompts=1024 \
  --tokenizer=${LOCAL_TOKENIZER_PATH}
```
