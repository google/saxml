## Offline Quantization tool
Because this is an experimental tool, it has not been integrated into the package or release process in a formalized manner. Therefore, you may have to install the dependencies in editable mode. This will be cleaned up in the future. Furthermore, it has only been tested for the DirectRunner (local machine), and not on GCP Dataflow. We hope to add this feature later.

TODO:
1. Clean up the build process for this tool.
2. Fix code to work on GCP Dataflow.

### Instructions
Make sure you've created a sax checkpoint of your model. It should be on your local machine. (You should also have the same copy in GCS when you want to use SAX for model serving later).

Clone the saxml, paxml and praxis repos all at the same directory level.

Your directory should look like:

```
- pwd
  - saxml
  - paxml
  - praxis
```


Next install the dependencies:

```
# Create a virtual environment
pyenv virtualenv 3.10.0 sax-quant
pyenv activate sax-quant

Install dependencies
pip install --upgrade pip
pip install apache-beam

cd paxml
pip install -e .
cd ..

cd praxis
pip install -e .
cd ..

pip install --upgrade --force-reinstall protobuf tensorflow lingvo
```
Then run the quantization tool

```
python saxml/saxml/tools/offline_quantize.py --input_dir path/to/input/checkpoint/checkpoint_00000000/state --output_dir path/to/output/checkpoint/checkpoint_00000000/state --quantization_configs "gptj"
```

This only generates files in `path/to/output/checkpoint/checkpoint_00000000/state`.
You will also need the other directories such as `metadata` and `descriptor`,
and the `commit_success.txt` files. To do this, you can do.

```
gsutil cp -r gs://mybucket/fp32/checkpoint_00000000 gs://mybucket/int8/
gsutil cp -r path/to/output/checkpoint/checkpoint_00000000/state gs://mybucket/int8/checkpoint_00000000
```