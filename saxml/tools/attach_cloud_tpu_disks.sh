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

PROJECT_ID=""
ZONE=""
WORKER_PREFIX=""
NUM_WORKERS=""
NAME=""

usage() {
  echo "Usage: $0 -p PROJECT_ID -z ZONE -w WORKER_PREFIX -n NUM_WORKERS NAME" 1>&2
  exit 1
}

while getopts "p:z:w:n:" options; do
  case "${options}" in
    p)
      PROJECT_ID=${OPTARG}
      ;;
    z)
      ZONE=${OPTARG}
      ;;
    w)
      WORKER_PREFIX=${OPTARG}
      ;;
    n)
      NUM_WORKERS=${OPTARG}
      ;;
    :)
      echo "Error: -${OPTARG} requires an argument"
      usage
      ;;
    *)
      usage
      ;;
  esac
done

shift "$((OPTIND-1))"

if [ "$#" -ne 1 ]; then
  usage
else
  NAME=$1
fi

if [ "${PROJECT_ID}" == "" ]; then
  echo "Error: missing PROJECT_ID"
  usage
fi

if [ "${ZONE}" == "" ]; then
  echo "Error: missing ZONE"
  usage
fi

if [ "${WORKER_PREFIX}" == "" ]; then
  echo "Error: missing WORKER_PREFIX"
  usage
fi

if [ "${NUM_WORKERS}" == "" ]; then
  echo "Error: missing NUM_WORKERS"
  usage
fi

if [ "${NAME}" == "" ]; then
  echo "Error: missing NAME"
  usage
fi

for (( i=0; i<${NUM_WORKERS}; i++ ))
do
  gcloud alpha compute disks create ${NAME}-disk-${i} --size=35 --type=pd-ssd --project=${PROJECT_ID} --zone=${ZONE}
done

for (( i=0; i<${NUM_WORKERS}; i++ ))
do
  gcloud alpha compute instances attach-disk ${WORKER_PREFIX}-w-${i} --project=${PROJECT_ID} --zone=${ZONE} --mode=rw --disk=${NAME}-disk-${i}
  gcloud alpha compute instances set-disk-auto-delete ${WORKER_PREFIX}-w-${i} --project=${PROJECT_ID} --zone=${ZONE} --auto-delete --disk=${NAME}-disk-${i}
done

gcloud alpha compute tpus tpu-vm ssh ${NAME} --project=${PROJECT_ID} --zone=${ZONE} --worker=all --command="sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb"

gcloud alpha compute tpus tpu-vm ssh ${NAME} --project=${PROJECT_ID} --zone=${ZONE} --worker=all --command="sudo mkdir -p /mnt/disks/persist && sudo mount -o discard,defaults /dev/sdb /mnt/disks/persist && sudo chmod 774 /mnt/disks/persist"

exit 0
