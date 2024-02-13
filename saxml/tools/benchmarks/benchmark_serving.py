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
r"""Benchmark online serving throughput.

It referrs to
https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py
"""

import argparse
import asyncio
import concurrent
import json
import random
import time
from typing import Any, AsyncGenerator, List, Tuple
import numpy as np
import sax
import sentencepiece
from tqdm.asyncio import tqdm

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []


def sample_requests(
    dataset_path: str,
    num_requests: int,
    max_input_len: int,
    max_output_len: int,
    tokenizer: Any,
) -> List[Tuple[str, int, int]]:
  """Sample the requests from the dataset."""
  if dataset_path is None:
    return [
        (tokenizer.decode([100] * max_input_len), max_input_len, max_output_len)
        for _ in range(num_requests)
    ]
  # Load the dataset.
  with open(dataset_path) as f:
    dataset = json.load(f)
  # Filter out the conversations with less than 2 turns.
  dataset = [data for data in dataset if len(data["conversations"]) >= 2]
  # Only keep the first two turns of each conversation.
  dataset = [
      (data["conversations"][0]["value"], data["conversations"][1]["value"])
      for data in dataset
  ]

  # Tokenize the prompts and completions.
  prompts = [prompt for prompt, _ in dataset]
  prompt_token_ids = tokenizer.encode(prompts)
  completions = [completion for _, completion in dataset]
  completion_token_ids = tokenizer.encode(completions)
  tokenized_dataset = []
  for i in range(len(dataset)):
    output_len = len(completion_token_ids[i])
    tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

  # Filter out too long sequences.
  filtered_dataset: List[Tuple[str, int, int]] = []
  for prompt, prompt_token_ids, output_len in tokenized_dataset:
    prompt_len = len(prompt_token_ids)
    if prompt_len < 4 or output_len < 4:
      # Prune too short sequences.
      # This is because TGI causes errors when the input or output length
      # is too short.
      continue
    if prompt_len > max_input_len or output_len > max_output_len:
      # Prune too long sequences.
      continue
    filtered_dataset.append((prompt, prompt_len, output_len))

  # Sample the requests.
  sampled_requests = random.sample(filtered_dataset, num_requests)
  return sampled_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
  """Get the requests from the input requests based on the rate."""
  input_requests = iter(input_requests)
  for request in input_requests:
    yield request

    if request_rate == float("inf"):
      # If the request rate is infinity, then we don't need to wait.
      continue
    # Sample the request interval from the exponential distribution.
    interval = np.random.exponential(1.0 / request_rate)
    # The next request will be sent after the interval.
    await asyncio.sleep(interval)


async def send_request(
    lm_client: Any,
    prompt: str,
    prompt_len: int,
    output_len: int,
    tokenizer: Any,
    pbar: tqdm,
) -> None:
  """Send the request."""
  loop = asyncio.get_running_loop()
  request_start_time = time.perf_counter()
  options = sax.ModelOptions()
  options.SetTimeout(3600)
  options.SetExtraInput("per_example_max_decode_steps", output_len)
  sax_response = await loop.run_in_executor(
      None, lm_client.Generate, prompt, options
  )
  output_token_ids = tokenizer.encode(sax_response[0][0])
  real_output_len = len(output_token_ids)
  request_end_time = time.perf_counter()
  request_latency = request_end_time - request_start_time
  REQUEST_LATENCY.append((prompt_len, real_output_len, request_latency))
  pbar.update(1)


async def benchmark(
    model: str,
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    tokenizer: Any,
) -> None:
  """Benchmark the online serving throughput."""
  tasks: List[asyncio.Task] = []
  pbar = tqdm(total=len(input_requests))
  model_client = sax.Model(model)
  lm_client = model_client.LM()
  loop = asyncio.get_running_loop()
  loop.set_default_executor(
      concurrent.futures.ThreadPoolExecutor(max_workers=400)
  )

  async for request in get_request(input_requests, request_rate):
    prompt, prompt_len, output_len = request
    task = asyncio.create_task(
        send_request(
            lm_client,
            prompt,
            prompt_len,
            output_len,
            tokenizer,
            pbar,
        )
    )
    tasks.append(task)
  await asyncio.gather(*tasks)
  pbar.close()


def main(args: argparse.Namespace):
  print(args)
  random.seed(args.seed)
  np.random.seed(args.seed)

  tokenizer = sentencepiece.SentencePieceProcessor(model_file=args.tokenizer)
  input_requests = sample_requests(
      args.dataset,
      args.num_prompts,
      args.max_input_length,
      args.max_output_length,
      tokenizer,
  )

  benchmark_start_time = time.perf_counter()
  asyncio.run(
      benchmark(
          args.model,
          input_requests,
          args.request_rate,
          tokenizer,
      )
  )
  benchmark_end_time = time.perf_counter()
  benchmark_time = benchmark_end_time - benchmark_start_time
  print(f"Total time: {benchmark_time:.2f} s")
  print(f"Throughput: {args.num_prompts / benchmark_time:.2f} requests/s")

  total_tokens = np.sum([output_len for _, output_len, _ in REQUEST_LATENCY])
  output_tokens_per_min = 60 * total_tokens / benchmark_time
  print(f"Output_tokens/min: {output_tokens_per_min:.2f}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Benchmark the online serving throughput."
  )
  parser.add_argument(
      "--model",
      type=str,
      default=None,
      help="publised model name in the sax cluster.",
  )
  parser.add_argument(
      "--dataset",
      type=str,
      help=(
          "Path to the dataset. If no dataset, generate the dummy prompt with"
          " length as max-input-length"
      ),
  )
  parser.add_argument(
      "--tokenizer",
      type=str,
      required=True,
      help="Name or path of the tokenizer.",
  )
  parser.add_argument(
      "--num-prompts",
      type=int,
      default=1024,
      help="Number of prompts to process.",
  )
  parser.add_argument(
      "--max-input-length",
      type=int,
      default=1024,
      help=(
          "Maximum number of input tokens for filtering the benchmark dataset."
      ),
  )
  parser.add_argument(
      "--max-output-length",
      type=int,
      default=1024,
      help=(
          "Maximum number of input tokens for filtering the benchmark dataset."
      ),
  )
  parser.add_argument(
      "--request-rate",
      type=float,
      default=float("inf"),
      help=(
          "Number of requests per second. If this is inf, "
          "then all the requests are sent at time 0. "
          "Otherwise, we use Poisson process to synthesize "
          "the request arrival times."
      ),
  )
  parser.add_argument("--seed", type=int, default=0)
  main(parser.parse_args())
