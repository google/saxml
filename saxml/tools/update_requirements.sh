#!/bin/bash

set -e

BASE=$(mktemp -d)

copybara $(dirname $0)/../copy.bara.sky to_folder_experimental .. --folder-dir=${BASE}/saxml

docker run -it -v ${BASE}/saxml:/saxml python:3.9 \
    bash -c "pip install -U pip pip-tools && cd /saxml && pip-compile --allow-unsafe requirements.in --output-file=requirements.txt && pip-compile --allow-unsafe requirements-cuda.in --output-file=requirements-cuda.txt"

cp ${BASE}/saxml/requirements*.txt $(dirname $0)/../