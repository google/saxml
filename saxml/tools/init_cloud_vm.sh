#!/bin/bash

set -e

sudo apt -y update && sudo apt install -y apt-transport-https curl gnupg patch python3-pip

curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key --keyring /usr/share/keyrings/bazel-archive-keyring.gpg add -
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt -y update && sudo apt install -y bazel

pip3 install numpy  # @org_tensorflow requires numpy installation in system