# Language model sampling decode parameters

This document summarizes the input parameters for language model with sample
decode.

## Parameters set in the template

Here are the parameters that is set in SAX language model
[template](https://github.com/google/saxml/blob/main/saxml/server/pax/lm/params/template.py).


| Parameter       | Usage                      | Extra information          |
| :-------------- | :--------------------------| :--------------------------|
| BATCH_SIZE      | Batch size of a request, it| Set it as 1 when           |
|                 | can be either an integer or| NUM_CACHE_SLOTS > 0.       |
|                 | a list. The input requests |                            |
|                 | will be padded to the      |                            |
|                 | nearest batch size during  |                            |
|                 | serving.                   |                            |
| TOP_K           | The upper bound of top_k   |                            |
|                 | used in sampling decode    |                            |
| NUM_SAMPLES     | Number of samples generated|                            |
|                 | for each prompt.           |                            |
| MAX_DECODE_STEPS| Maximum decoding steps, it | When it is a sequence, the |
|                 | can be either an integer or| sampling decode will have  |
|                 | a sequence. used in        | multiple decoding loops,   |
|                 | sampling decode            | each with different        |
|                 |                            | sequence length in the     |
|                 |                            | decoding cache.            |
| INPUT_SEQ_LEN   | Maximum input prefix       | If BUCKET_KEYS is set, the |
|                 | length.                    | BUCKET_KEYS should be      |
|                 |                            | smaller than or equal to   |
|                 |                            | INPUT_SEQ_LEN.             |
| BUCKET_KEYS     | Input sequence length      |                            |
|                 | buckets. For example, if   |                            |
|                 | INPUT_SEQ_LEN is set to    |                            |
|                 | 2048, BUCKET_KEYS can be   |                            |
|                 | set to [512, 1024, 2048].  |                            |
|                 | Input sequence will be     |                            |
|                 | padded to the next larger  |                            |
|                 | length in the BUCKET_KEYS. |                            |
| EXTRA_INPUTS    | A map of extra input for   | More information can be    |
|                 | each request. It is a      | found in the next section  |
|                 | dictionary of {string\:    |                            |
|                 | float }.                   |                            |
| EOS_ID, SOS_ID  | End/start of sequence id   |                            |
|                 | for the tokenizer.         |                            |
| STOP_TOKEN_IDS  | Extra token ids to stop    |                            |
|                 | decoding. Default is       |                            |
|                 | [EOS_ID]. It needs to      |                            |
|                 | include EOS_ID in the list |                            |
|                 | explicitly.                |                            |
| NUM_CACHE_SLOTS | Number of slot for storing | Experimental Support.      |
|                 | the KV Cache for next token|                            |
|                 | generation. When it > 0,   |                            |
|                 | the continuous batching    |                            |
|                 | will be enabled.           |                            |
|                 | It becomes the             |                            |
|                 | batch size for             |                            |
|                 | auto-regressive            |                            |
|                 | decoding process.          |                            |

## Per request parameters

These input parameters can be configured with
[extra_inputs](https://github.com/google/saxml/blob/ab36fc08ed921682cccb796a79c1d69dd1b23074/saxml/protobuf/lm.proto#L33-L37).

Parameter                    | Usage                                                                                                                                                                                                                                            | Extra information
:--------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------
temperature                  | To set the tempeature of sampling decode. The temperature could be a scalar for per request temperature or a 1D tensor for per sample temperature. When setting the per sample temperature, the 1D tensor's size should be equal to num_samples. | Could use `options.SetExtraInputTensor()` to set per sample temperature.
per_example_top_k            | To set the top_k used in sampling decode                                                                                                                                                                                                         | Needs be smaller than TOP_K.
per_example_top_p            | To set the top_p used in sampling decode                                                                                                                                                                                                         | When both per_example_top_k and per_example_top_p are defined, top_k will be applied before top_p.
per_example_max_decode_steps | Maximum decoding steps for each requests. Needs to be smaller than maximum value of MAX_DECODE_STEPS                                                                                                                                             |
eos_id                       | To set the stop sequence, it needs to be a 1D tensor with fixed length. Should be padded with 0s on the left if the end of sequence length is smaller than the fixed length.                                                                     | Setting 'eos_id' as scalar, for example {'eos_id': 1} is not supported. It is suggested to use STOP_TOKEN_IDS = [1, 7, ...] if users would like to have different stop tokens.

## FAQ

### I tried adding extra input in my request, but it doesn't work.

For example, you might have this error message:

`StatusNotOk: rpc error: code = InvalidArgument desc = key eos_id in RPC
request's extra_inputs field is not in ServableModel.extra_inputs.extra_inputs
in ServableModel are {'temperature': 0.1, 'per_example_max_decode_steps': 512,
'per_example_top_k': 40} [INVALID_ARGUMENT]`

The reason for the error is the model configuration extra_input doesn't have
`eos_id` as key. It needs to add {'eos_id': [1, 7]} in the model configuration.

### How to set parameter in the template during publish?
Example using `saxutil`:

```shell
$ ./saxutil publish /sax/test/test $MODEL_CONFIG_CLASS $CHECKPOINT 1 BATCH_SIZE=32 NUM_SAMPLES=1 BUCKET_KEYS=[256,512,1024] TOP_K=1 INPUT_SEQ_LEN=1024 MAX_DECODE_STEPS=1024 EXTRA_INPUTS={\"temperature\":0\,\"per_example_max_decode_steps\":1024\,\"per_example_top_k\":1\,\"per_example_top_p\":0}
```

### How to set extra input for each request?

Example using `saxutil`:

```shell
$ saxutil lm.generate -extra="per_example_max_decode_steps:32,temperature:0" /sax/test/test "Hello"
```

Example in the colab:

```python
lm_model = '/sax/test/test'
model_path, ckpt_path, _ = sax.List(sax_model_b)
lm = sax.Model(lm_model).LM()

options = sax.ModelOptions()
options.SetExtraInput("temperature", 0.0)
options.SetExtraInput("per_example_max_decode_steps", 256)
options.SetTimeout(80)
sax_results = lm.Generate("Hello", options)
```
