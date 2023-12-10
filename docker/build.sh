#!/bin/bash
set -e
set -u
SCRIPTROOT="$( cd "$(dirname "$0")" ; pwd -P )"
cd "${SCRIPTROOT}/.."

docker build -t ubuntu20 -f docker/Dockerfile .