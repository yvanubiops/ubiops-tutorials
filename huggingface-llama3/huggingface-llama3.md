# Deploying Llama 3 8B to UbiOps

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/blob/master/docs/huggingface-llama3/huggingface-llama3/huggingface-llama3.ipynb){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/huggingface-llama3/huggingface-llama3/huggingface-llama3.ipynb){ .md-button .md-button--secondary }

This notebook will show you how you can create a cloud-based inference API endpoint for the 
[Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model using UbiOps. The Llama model 
is already pre-trained and will be loaded from the Huggingface [meta-llama](https://huggingface.co/meta-llama) library. 
Note that downloading this model requires you to have a Hugginface token that has sufficient permissionsto download Llama 3.

The [Meta Llama 3 models](https://llama.meta.com/llama3/) are a collection of pre-trained and instruction tuned generative 
text models, in 8B and 70B parameter sizes. The instruction versions of these models are optimized for dialogue use cases. 
Meta claims that Llama 3B outperforms models with a similar size, like Mistral 7B & Gemma 7B on common industry benchmarks. 
The model deployed in this tutorial is the instruction tuned version of the Llama 8B model. We also optimize the inference 
speed using the [flash attention library](https://github.com/Dao-AILab/flash-attention).

In this notebook, we will walk you through:

1. Connecting with the UbiOps API client
2. Creating a code environment for our deployment
3. Creating a deployment for the Llama-3-8B-Instruct
4. Calling the Llama 3 deployment endpoint

Llama 3 is a text-to-text model. Therefore we will make a deployment that takes a text prompt as input, and returns a 
response. Next to the user's input, we will also add the `system_prompt` and `config` to the deployment's input. Using
this set-up enables you to experiment with different system prompts and generation parameters to see how they affect the 
responses of the model.


The deployment will return the `input`, which is the user's `prompt` and `system_prompt`, and the `used_config`.

Default pre-set values will be used for the `system_prompt` and `config` if these are not provided, these can be found
in the `__init__` statement of the `deployment.py`.

|Deployment input & output variables| **Variable name** |**Data type** |
|--------------------|--------------|--------------|
| **Input fields**   | prompt | string |
|                    | system_prompt | string |
|                    | config | dictionary|
| **Output fields**  | output | string |
|                    | input        | string |
|                    | used_config  | dictionary |

Note that we deploy to a GPU instance by default, which is not accessible in every project. You can 
[contact us](https://ubiops.com/contact-us/) about this.

Let's start coding!


## 1. Set up a connection with the UbiOps API client

To use the UbiOps API from our notebook, we need to install the UbiOps Python Client Library first:


```python
!pip install -qU ubiops
```

Now we can set up a connection with your UbiOps environment. To do this we will need the name of your UbiOps project and 
an API token with the `project_editor` permissions.

You can paste your project name and API token in the code block below before running.


```python
import ubiops
from datetime import datetime

API_TOKEN = "<INSERT API_TOKEN WITH PROJECT EDITOR RIGHTS>"  # Make sure this is in the format "Token token-code"
PROJECT_NAME = "<INSERT PROJECT NAME IN YOUR ACCOUNT>"

HF_TOKEN = "<INSERT HUGGINGFACE TOKEN WITH CORRECT ACCESS>"  # We need this token to download the model from Huggingface 

DEPLOYMENT_NAME = f"llama-3-8b-{datetime.now().date()}"
DEPLOYMENT_VERSION = "v1"

# Initialize client library
configuration = ubiops.Configuration(host="https://api.ubiops.com/v2.1")
configuration.api_key["Authorization"] = API_TOKEN

# Establish a connection
client = ubiops.ApiClient(configuration)
api = ubiops.CoreApi(client)
print(api.projects_get(PROJECT_NAME))
```

## 2. Setting up the environment

[The environment](https://ubiops.com/docs/environments/#environments) that our model runs in can be managed separately. To do this we need to select a base environment, to which
we will add additional dependencies.


```python
environment_dir = "environment_package"
ENVIRONMENT_NAME = "llama-3-environment"
```


```python
%mkdir {environment_dir}
```

We will define the Python packages required to run the model in a `requirements.txt`, which we will later upload to UbiOps. 


```python
%%writefile {environment_dir}/requirements.txt
# This file contains package requirements for the environment
# Installed via PIP.
torch==2.0.1+cu118
huggingface-hub==0.20.3
transformers==4.38.1
scipy
diffusers
safetensors
ninja
jupyterlab==4.0.11
notebook==7.0.7
ipywidgets
https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Now we will create a `ubiops.yaml` to set a remote pip index. This ensures that we will install a CUDA-compatible version
of PyTorch. CUDA allows models to be loaded and to run on GPUs.


```python
%%writefile {environment_dir}/ubiops.yaml
environment_variables:
- PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118
```

Now we create a custom environment on UbiOps. We select `Ubuntu 22.04 + Python 3.10 + CUDA 11.7.1` as the `base_environment`,
and add the additional dependencies we defined earlier to the `base_environment` to create a `custom environment`. The
environment will be called `llama-3-environment`.


```python
api_response = api.environments_create(
    project_name=PROJECT_NAME,
    data=ubiops.EnvironmentCreate(
        name=ENVIRONMENT_NAME,
        # display_name=ENVIRONMENT_NAME,
        base_environment="ubuntu22-04-python3-10-cuda11-7-1", 
        description="Environment to run Llama-3 from Huggingface",
    ),
)
```

Package and upload the environment files.


```python
import shutil

training_environment_archive = shutil.make_archive(
    environment_dir, "zip", ".", environment_dir
)
api.environment_revisions_file_upload(
    project_name=PROJECT_NAME,
    environment_name=ENVIRONMENT_NAME,
    file=training_environment_archive,
)
```

## 3. Creating a deployment for the Llama 3 8B Instruct model

With the environment set up, we can start writing the code to run the Llama-3-8B model, and push it to UbiOps.

We will create a `deployment.py` with a `Deployment` class, which has two methods:

- The `__init__`which will run when the deployment starts up. This method can be used to load models, data artefacts and 
other requirements for inference.
- The `request()` will run every time a call is made to the models REST API endpoint and includes all the logic for 
processing data.

Separating the logic between the two methods will ensure fast model response times. The model will be loaded in the 
`__init__` method, and the code that needs to be run when a call is made to the deployment in the `request()` method.
This way the model only needs to be loaded in when the deployment starts up.

As mentioned in the introduction, we will add a default `system_prompt` and `config` to the input. 


```python
deployment_code_dir = "deployment_code"
```


```python
!mkdir {deployment_code_dir}
```


```python
%%writefile {deployment_code_dir}/deployment.py
import os
import torch
import shutil
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    LlamaForCausalLM, 
    GenerationConfig
)
from huggingface_hub import login



class Deployment:
    def __init__(self, base_directory, context):
        """
        Initialisation method for the deployment. Any code inside this method will execute when the deployment starts up.
        It can for example be used for loading modules that have to be stored in memory or setting up connections.
        """

        print("Initialising deployment")
        
        # Read out model-related environment variables
        LLAMA_VERSION = os.environ.get('LLAMA_VERSION', 'meta-llama/Meta-Llama-3-8B-Instruct')
        self.REPETITION_PENALTY = float(os.environ.get('REPETITION_PENALTY', 1.15))
        self.MAX_RESPONSE_LENGTH  = float(os.environ.get('MAX_RESPONSE_LENGTH', 256))
        
        # Login to Huggingface
        HF_TOKEN = os.environ["HF_TOKEN"]
        login(token=HF_TOKEN)

   
        print("Downloading tokenizer")
        
        self.tokenizer = AutoTokenizer.from_pretrained(LLAMA_VERSION, 
                                                       device_map = 'auto'
        )
        self.model = LlamaForCausalLM.from_pretrained(LLAMA_VERSION, 
                                                      torch_dtype = torch.float16, 
                                                      device_map = 'auto', 
                                                      use_safetensors = True,
                                                      attn_implementation="flash_attention_2",
        )

  

        self.pipe = pipeline(
            os.environ.get("PIPELINE_TASK", "text-generation"),
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
        )
        
        # Set default prompt generation variables
        self.messages = [
            {"role": "system", "content": "{system_prompt}"},
            {"role": "user", "content": "{user_prompt}"},
        ]
        

        self.terminators = [
            self.pipe.tokenizer.eos_token_id,
            self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        
        
        self.default_config = {
            'do_sample': True,
            'max_new_tokens': self.MAX_RESPONSE_LENGTH,
            'temperature': 0.6,
        }
        
        
    def request(self, data):
        """
        Method for deployment requests, called separately for each individual request.
        """
        print("Processing request")
        
        if data["system_prompt"]:
            system_prompt = data["system_prompt"]
        else:
            system_prompt = "You are a pirate chatbot who always responds in pirate speak!"
            
        config = self.default_config.copy()
        
        # Update config dic if the user added a config dict
        if data["config"]:
            config.update(data["config"])
            
        # Create the full prompt
        formatted_messages = []
        for message in self.messages:
            # Use format() method to format the content of each dictionary
            formatted_content = message["content"].format(
                system_prompt=system_prompt, user_prompt=data["prompt"]
            )
            # Append the formatted content to the new list
            formatted_messages.append({"role": message["role"], "content": formatted_content})

        full_prompt = self.pipe.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        print(f"The input for the model is this full prompt \n:{full_prompt}")
        
        # Generate text
        sequences = self.pipe(
            full_prompt,
            eos_token_id=self.terminators,
            **config
        )

        response = sequences[0]["generated_text"]

        # Here we set our output parameters in the form of a json
        return {"output": response, 
                "input": full_prompt, 
                "used_config": config
        }

```

### Create a UbiOps deployment

Now we can create the deployment, where we define the in- and outputs of the model. Each deployment can have multiple versions.
For each, version you can use a different deployed code, environment, instance type, among other settings.


```python
# Create the deployment
deployment_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME,
    input_type="structured",
    output_type="structured",
    input_fields=[
        {"name": "prompt", "data_type": "string"},
        {"name": "system_prompt", "data_type": "string"},
        {"name": "config", "data_type": "dict"},
    ],
    output_fields=[
        {"name": "output", "data_type": "string"},
        {"name": "input", "data_type": "string"},
        {"name": "used_config", "data_type": "dict"},
    ],
)

api.deployments_create(project_name=PROJECT_NAME, data=deployment_template)
```

### Create a deployment version

Now we will create a version of the deployment. For the version, we need to define the name, environment, instance type 
(CPU or GPU) as well as the size of the instance.

For this model it is recommended to use a GPU instance.


```python
# Create the version
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment=ENVIRONMENT_NAME,
    instance_type="16384mb_l4",
    maximum_instances=1,
    minimum_instances=0,
    maximum_idle_time=600,  # = 10 minutes
    request_retention_mode="full",
)

api.deployment_versions_create(
    project_name=PROJECT_NAME, deployment_name=DEPLOYMENT_NAME, data=version_template
)
```

Package and upload the code:


```python
import shutil

deployment_code_archive = shutil.make_archive(
    deployment_code_dir, "zip", deployment_code_dir
)

upload_response = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    file=deployment_code_archive,
)
print(upload_response)

# Check if the deployment is finished building. This can take a few minutes
ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    revision_id=upload_response.revision,
    stream_logs=True,
)
```

Before we can send requests to our deployment version, the environment has to be finished building. Note that building the
 environment might take a while  as UbiOps needs to download and install all the packages and dependencies. The environment
 only needs to be built once, the next time that an instance type is spun up for our deployment the dependencies do not have
 to be installed anymore. You can toggle off `stream_logs` to not stream logs of the build process.

### Create an environment variable

Here we create environment variables for the Huggingface token. We need this token to allow us to download the Llama model
from Huggingface, since it's behind a gated repo. 

If you want to use a different version of Llama 3, you can also add an environment variable for the `model_id` by adding 
this code to the code cell below:

<details>
 <summary>Click here to see the code that creates an environment variable for the `model_id`</summary>

```python
MODEL_ID = "ENTER THE MODEL_ID HERE"  # You can change this parameter if you want to use a different model from Huggingface.


api_response = api.deployment_version_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    data=ubiops.EnvironmentVariableCreate(
        name="model_id", value=MODEL_ID, secret=False
    ),
)
```
</details>


```python
api_response = api.deployment_version_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    data=ubiops.EnvironmentVariableCreate(name="model_id", value=HF_TOKEN, secret=True),
)
```

## 4. Calling the Llama 3 8B deployment API endpoint

Our deployment is now ready to process requests! We can send requests to the deployment using either the 
[`deployment-requests-create` or `batch-deployment-requests-create`](https://ubiops.com/docs/requests/#request-types) 
API endpoint. During this step a node will be spun up, and the model will be downloaded from Huggingface. Hence why this 
step can take a while. You can monitor the progress of the process in the [logs](https://ubiops.com/docs/monitoring/logging/). 
Subsequent requests to the deployment will be handled faster. 

### Make a request using the default `system_prompt` and `config`.


```python
data = {"prompt": "tell me a joke", "system_prompt": "", "config": {}}

api.deployment_requests_create(
    project_name=PROJECT_NAME, deployment_name=DEPLOYMENT_NAME, data=data, timeout=3600
).result
```

### Make a request using other values for the `system_prompt` and `config`.

For this request, we will instruct the LLM to translate English texts into the style of Shakespearean. We will let the model
be more creative with generating sequences by lowering the `temperature` parameter. The text used for this example is shown
in the cell below:


```python
text = "In the village of Willowbrook lived a girl named Amelia, known for her kindness and curiosity. One autumn day, she ventured into the forest and stumbled upon an old cottage filled with dusty tomes of magic. Amelia delved into the ancient spells, discovering her own hidden powers. As winter approached, a darkness loomed over the village. Determined to protect her home, Amelia confronted the source of the darkness deep in the forest. With courage and magic, she banished the shadows and restored peace to Willowbrook., Emerging triumphant, Amelia returned home, her spirit ablaze with newfound strength. From that day on, she was known as the brave sorceress who saved Willowbrook, a legend of magic and courage that echoed through the ages."
```


```python
data = {
    "prompt": text,
    "system_prompt": "You are a friendly chatbot that translates texts into the style of Shakespearean.",
    "config": {
        "do_sample": True,
        "max_new_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.5,
    },
}

api.deployment_requests_create(
    project_name=PROJECT_NAME, deployment_name=DEPLOYMENT_NAME, data=data, timeout=3600
).result
```

So that's it! You now have your own on-demand, scalable Llama-3-8B-Instruct-v0.2 model running in the cloud, with a REST API that you can reach from anywhere!
