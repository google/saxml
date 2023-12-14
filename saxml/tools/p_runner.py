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
"""Parallel runner."""
import concurrent.futures


class PRunner:
  """Runs callback in parallel."""

  def __init__(self, callback, num_threads):
    """Constructor.

    Args:
      callback: A callback function. Each callback is a function of signature
        worker_fn(list_of_actions) -> dict(...)
      num_threads: A Python int.
    """
    self.callback = callback
    self.num_threads = num_threads

  def run(self, actions, verbose=False):
    """Run."""
    if verbose:
      print(f'Num actions: {len(actions)}', flush=True)
    num_threads = self.num_threads
    callback = self.callback

    futures = []
    actions_per_thread = (len(actions) + num_threads - 1) // num_threads

    final_res = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_threads) as executor:
      # Start the load operations and mark each future with its URL
      for i in range(num_threads):
        start_idx = i * actions_per_thread
        end_idx = min(start_idx + actions_per_thread, len(actions))
        actions_slice = actions[start_idx:end_idx]
        if actions_slice:
          futures.append(executor.submit(callback, actions_slice))

      for future in concurrent.futures.as_completed(futures):
        try:
          res = future.result()
          final_res.update(res)
        except Exception as exc:  # pylint:disable=broad-except
          print(exc, flush=True)

    if verbose:
      print('Done future', flush=True)
    return final_res
