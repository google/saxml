#!/bin/bash
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -e

sudo apt -y update && sudo apt install -y apt-transport-https curl gnupg patch python3-pip git

curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key --keyring /usr/share/keyrings/bazel-archive-keyring.gpg add -
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt -y update && sudo apt install -y bazel

pip3 install -U pip
pip3 install numpy  # @org_tensorflow requires numpy installation in system