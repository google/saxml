# Saxml (aka Sax)

Saxml is an experimental system that serves
[Paxml](https://github.com/google/paxml) models for inference.

A Sax cluster consists of an admin server and a group of model servers.
The admin server keeps track of model servers, assigns published models to
model servers to serve, and helps clients locate model servers serving
specific published models.

## Install Sax

### Install and set up the `gcloud` tool

[Install](https://cloud.google.com/sdk/gcloud#download_and_install_the) the
`gcloud` CLI and set the default account and project:

```
gcloud config set account <your-email-account>
gcloud config set project <your-project>
```

### Create a Cloud Storage bucket to store Sax server states

[Create](https://cloud.google.com/storage/docs/creating-buckets) a
Cloud Storage bucket:

```
gcloud storage buckets create gs://sax-data
```

### Create a Compute Engine VM instance for the admin server

[Create](https://cloud.google.com/compute/docs/create-linux-vm-instance) a
Compute Engine VM instance:

```
gcloud compute instances create sax-admin \
  --zone=us-central1-b \
  --machine-type=e2-standard-8 \
  --boot-disk-size=1TB \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

### Create a Cloud TPU VM instance for a model server

Use this [guide](https://cloud.google.com/tpu/docs/users-guide-tpu-vm) to
enable the Cloud TPU API in a Google Cloud project.

Create an 8-core Cloud TPU VM instance:

```

gcloud compute tpus tpu-vm create sax-tpu \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --version=tpu-vm-v4-base \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

### Start the Sax admin server

SSH to the Compute Engine VM instance:

```
gcloud compute ssh --zone=us-central1-b sax-admin
```

Inside the VM instance, clone the Sax repo and initialize the environment:

```
git clone https://github.com/google/saxml.git
cd saxml
saxml/tools/init_cloud_vm.sh
```

Configure the Sax admin server. This only needs to be done once:

```
bazel run saxml/bin:admin_config -- \
  --sax_cell=/sax/test \
  --sax_root=gs://sax-data/sax-root \
  --fs_root=gs://sax-data/sax-fs-root \
  --alsologtostderr
```

Start the Sax admin server:

```
bazel run saxml/bin:admin_server -- \
  --sax_cell=/sax/test \
  --sax_root=gs://sax-data/sax-root \
  --port=10000 \
  --alsologtostderr
```

### Start the Sax model server

SSH to the Cloud TPU VM instance:

```
gcloud compute tpus tpu-vm ssh --zone=us-central2-b sax-tpu
```

Inside the VM instance, clone the Sax repo and initialize the environment:

```
git clone https://github.com/google/saxml.git
cd saxml
saxml/tools/init_cloud_vm.sh
```

Start the Sax model server:

```
bazel run saxml/server/server -- \
  --sax_cell=/sax/test \
  --sax_root=gs://sax-data/sax-root \
  --port=10001 \
  --platform_chip=tpuv4 \
  --platform_topology=1x1x8 \
  --alsologtostderr
```

## Use Sax

Sax comes with a command-line tool called `saxutil` for easy usage:

```
# From the `saxml` repo root directory:
alias saxutil='bazel run saxml/bin:saxutil -- --sax_root=gs://sax-data/sax-root'
```

`saxutil` supports the following commands:

- `saxutil help`: Show general help or help about a particular command.
- `saxutil ls`: List all cells, all models in a cell, or a particular model.
- `saxutil publish`: Publish a model.
- `saxutil unpublish`: Unpublish a model.
- `saxutil update`: Update a model.
- `saxutil lm.generate`: Use a language model generate suffixes from a prefix.
- `saxutil lm.score`: Use a language model to score a prefix and suffix.
- `saxutil lm.embed`: Use a language model to embed text into a vector.
- `saxutil vm.generate`: Use a vision model to generate images from text.
- `saxutil vm.classify`: Use a vision model to classify an image.
- `saxutil vm.embed`: Use a vision model to embed an image into a vector.

As an example, publish a Sax model by loading a
[Paxml model](https://github.com/google/paxml/blob/6627c5b4a1c9da5d7b4e87416cdf82e8cd0e9072/paxml/tasks/lm/params/lm_cloud.py#L164)
checkpoint:

```
saxutil publish \
  /sax/test/lm2b
  saxml.server.pax.lm.cloud.params.LmCloudSpmd2B \
  gs://sax-data/<checkpoint_path> \
  1
```

When the model is loaded, issue a query:

```
saxutil lm.generate /sax/test/lm2b "Q: Who is Harry Porter's mother? A: "
```

The result may be this:

```
+--------------+-----------+
|    SAMPLE    |   SCORE   |
+--------------+-----------+
| Lily Potter. | -5.878906 |
| Lily.        | -9.015625 |
| Lily Evans.  | -9.878906 |
| Mrs. Porter. | -8.265625 |
+--------------+-----------+
```
