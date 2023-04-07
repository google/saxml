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

FROM python:3.10

ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL True

RUN set -e

RUN apt -y update && apt install -y apt-transport-https curl gnupg patch python3-pip

RUN curl https://bazel.build/bazel-release.pub.gpg | apt-key --keyring /usr/share/keyrings/bazel-archive-keyring.gpg add -
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN apt -y update && apt install -y bazel-5.4.0 google-cloud-cli
RUN update-alternatives --install /usr/bin/bazel bazel /usr/bin/bazel-5.4.0 20

RUN pip3 install -U pip numpy

COPY saxml saxml
WORKDIR saxml

RUN bazel build saxml/bin:admin_config
RUN bazel build saxml/bin:admin_server
RUN bazel build saxml/bin:saxutil
RUN bazel build saxml/server:server

ENV SAX_ROOT /tmp/sax-root
ENV JAX_PLATFORMS cpu
RUN echo '#/bin/bash\n\
bazel-bin/saxml/bin/admin_config_/admin_config -sax_cell=/sax/test -sax_root=$SAX_ROOT -fs_root=/tmp/sax-fs-root\n\
bazel-bin/saxml/bin/admin_server_/admin_server -sax_cell=/sax/test &\n\
sleep 5\n\
bazel-bin/saxml/bin/saxutil_/saxutil publish /sax/test/ulm saxml.server.pax.lm.experimental.params.hybrid.HybridV3Decoder1x1x1Test gs://fake 1\n\
bazel-bin/saxml/server/server -sax_cell=/sax/test -port=10001 -platform_chip=cpu -platform_topology=1\n'\
> entrypoint.sh && chmod +x entrypoint.sh

ENTRYPOINT ./entrypoint.sh