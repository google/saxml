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
"""The module to expose the common functions to public."""

from saxml.server.pax.lm import lm_tokenizer
from saxml.server.pax.lm import servable_lm_common
from saxml.server.pax.lm.params import template


decode_fetch_output = servable_lm_common.decode_fetch_output
decode_tf_post_processing = servable_lm_common.decode_tf_post_processing
LMTokenizer = lm_tokenizer.LMTokenizer
make_servable = template.make_servable
