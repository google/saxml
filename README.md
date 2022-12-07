Sax-ml is system that serves [Pax-ml](https://github.com/google/paxml) models
for inference.

A Sax cluster is composed of an admin server and a group of model servers. The
admin server keeps track of model servers, assigns published models to model
servers to serve, and helps clients locate model servers serving specific
published models.

## Installation

The following is a guide for setting up a Sax cluster on Google Cloud Platform.

1)
[Set up a GCE VM and attach a TPU](https://cloud.google.com/tpu/docs/users-guide-tpu-vm).

2) Install Git, Python 3.9 and other dependencies

```
sudo apt-get update
sudo apt-get install -y curl less git python3-pip rsync gnupg emacs unzip python3.9 python3.9-distutils google-cloud-sdk
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1000
```

3) Install [Bazel](https://bazel.build/)
<!--* pragma: { seclinter_this_is_fine: true } *-->

```
sudo apt install apt-transport-https curl gnupg
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
sudo mv bazel-archive-keyring.gpg /usr/share/keyrings
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update && sudo apt install bazel
sudo apt update && sudo apt full-upgrade
```
<!--* pragma: { seclinter_this_is_fine: false } *-->
4) Clone this repo: `git clone https://github.com/google/saxml.git`

5) Build Sax: `cd saxml; bazel build ...`

## Usage

To run the saxutil binary, use `bazel run saxutil` or build and alias the binary
path.

Sax supports a wide range of models and saxutil provides the full list of
commands needed to interacts with those models. Here is a list of commands.

-   `saxutil help`: Show general help information or help information about a
    command.
-   `saxutil create`: Create a new sax cell.
-   `saxutil ls`: List all cells, or all published model in a cell, or details
    of a particular model in a cell.
-   `saxutil publish`: Publishes a model.
-   `saxutil unpublish`: Unpublishes a model.
-   `saxutil update`: Updates a model.
-   `saxutil am.recognize`: Invoke an ASR model to transcribe the given audio.
-   `saxutil lm.score`: Invoke a language model to score the given prefix and
    suffix.
-   `saxutil lm.generate`: Invoke a language model to generate a few suffixes
    given the prefix.
-   `saxutil lm.embed`: Invoke a language model to embed a text into a vector.
-   `saxutil vm.classify`: Invoke a vision model to classify an image (bytes).
-   `saxutil vm.generate`: Invoke a vision model to generate images for a text
    input.
-   `saxutil vm.embed`: Invoke a vision model to embed an image (bytes) into a
    vector.

As an example, suppose there is a working language model, a Q/A type of query
would look like the following:

```shell
$ saxutil lm.generate /sax/bar/model64b "Q: Who is Harry Porter's mother? A:"

$
+--------------+-----------+
|    SAMPLE    |   SCORE   |
+--------------+-----------+
| Lily Potter. | -5.878906 |
| Lily.        | -9.015625 |
| Lily Evans.  | -9.878906 |
| Mrs. Porter. | -8.265625 |
+--------------+-----------+
```
