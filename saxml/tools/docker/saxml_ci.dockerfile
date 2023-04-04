# Container used for ci testing.
#
# Exmaple build and run unit tests. Run from the root of the repo.
# $ docker build --pull --no-cache  --tag tensorflow:saxml_ci \
#     -f saxml/tools/docker/saxml_ci.dockerfile saxml/tools/docker/
#
# $ docker run -it --rm --mount "type=bind,src=$(pwd),dst=/tmp/saxml" \
#      --workdir="/tmp/saxml" tensorflow:saxml_ci  bash saxml/tools/ci_build.sh
#
FROM ubuntu:20.04

LABEL maintainer="no-reply@google.com"
ARG python_version="python3.8 python3.9 python3.10"
ARG APT_COMMAND="apt-get -o Acquire::Retries=3 -y"

# Newer versions of Bazel do not work when running as root.
ARG USERNAME=saxml
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Stops tzdata from asking about timezones and blocking install on user input.
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

# Installs basics including add-apt.
RUN ${APT_COMMAND} update && ${APT_COMMAND} install -y --no-install-recommends \
        software-properties-common \
        curl \
        less \
        unzip \
        sudo \
        vim \
        git-all

# Adds repository to pull versions of python from.
RUN add-apt-repository ppa:deadsnakes/ppa

# Pick up some TF dependencies
RUN ${APT_COMMAND} update && ${APT_COMMAND} install -y --no-install-recommends \
        build-essential \
        python3.9-dev \
        python3.10-dev \
        # python >= 3.8 needs distutils for packaging.
        python3.9-distutils \
        python3.10-distutils \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py

# Installs known working version of bazel.
ARG bazel_version=5.4.0
ENV BAZEL_VERSION ${bazel_version}
RUN mkdir /bazel && \
    cd /bazel && \
    curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

ARG pip_dependencies=' \
      numpy'

# Installs python dependencies for all version of python requested.
RUN for python in ${python_version}; do \
    $python get-pip.py && \
    $python -mpip --no-cache-dir install $pip_dependencies; \
  done
RUN rm get-pip.py

# Creates a non-root user with sudo access.
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

WORKDIR "/tmp/saxml"

CMD ["/bin/bash"]
