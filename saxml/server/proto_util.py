# Copyright 2022 Google LLC
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
"""Protobuf utilities."""

from typing import Optional

from saxml.protobuf import admin_pb2


def to_chip_type(chip: Optional[str]) -> admin_pb2.ModelServer.ChipType:
  """Returns the protobuf value for the given chip name string."""
  default = admin_pb2.ModelServer.ChipType.CHIP_TYPE_UNKNOWN
  if chip is None:
    return default
  chip_map = {
      'tpuv2': admin_pb2.ModelServer.ChipType.CHIP_TYPE_TPU_V2,
      'tpuv3': admin_pb2.ModelServer.ChipType.CHIP_TYPE_TPU_V3,
      'tpuv4': admin_pb2.ModelServer.ChipType.CHIP_TYPE_TPU_V4,
      'tpuv4i': admin_pb2.ModelServer.ChipType.CHIP_TYPE_TPU_V4I,
      'p100': admin_pb2.ModelServer.ChipType.CHIP_TYPE_GPU_P100,
      'v100': admin_pb2.ModelServer.ChipType.CHIP_TYPE_GPU_V100,
      'a100': admin_pb2.ModelServer.ChipType.CHIP_TYPE_GPU_A100,
      'cpu': admin_pb2.ModelServer.ChipType.CHIP_TYPE_CPU,
  }
  return chip_map.get(chip.lower(), default)


# TODO(jiawenhao): Add regex matching to handle more string variaties.
def to_chip_topology(
    topology: Optional[str],
) -> admin_pb2.ModelServer.ChipTopology:
  """Returns the protobuf value for the given topology description string."""
  default = admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_UNKNOWN
  if topology is None:
    return default
  if topology.endswith('_twisted'):
    topology = topology[: -len('_twisted')]
  if topology.endswith('_untwisted'):
    topology = topology[: -len('_untwisted')]
  topology_map = {
      '1': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_1,
      '2': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_2,
      '4': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_4,
      '8': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_8,
      '16': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_16,
      '1x1': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_1X1,
      '2x2': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_2X2,
      '4x2': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_4X2,
      '4x4': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_4X4,
      '4x8': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_4X8,
      '8x8': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_8X8,
      '8x16': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_8X16,
      '16x16': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_16X16,
      '16x32': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_16X32,
      '32x32': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_32X32,
      '1x1x1': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_1X1X1,
      '1x2x1': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_1X2X1,
      '2x2x1': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_2X2X1,
      '2x2x2': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_2X2X2,
      '2x2x4': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_2X2X4,
      '2x4x4': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_2X4X4,
      '4x4x4': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_4X4X4,
      '4x4x8': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_4X4X8,
      '4x4x16': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_4X4X16,
      '4x8x8': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_4X8X8,
      '8x8x12': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_8X8X12,
      '8x8x8': admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_8X8X8,
  }
  return topology_map.get(topology, default)
