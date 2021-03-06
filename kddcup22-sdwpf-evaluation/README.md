
# Manual on Evaluation


## Environment Setup   

1. System

   * OS: Ubuntu 18.04 LTS
   * GPU (one card)
     * Tesla V100 (32 GB)
     * CUDA: 11.2, CuDNN: 8
     * Driver: 460.32.03

3. Python version

    python = 3.7.11

4. Preinstalled Packages

   1. [base](preinstalled-pkgs/base_env_installed_packages.md)     (Updated)
   2. [paddlepaddle](preinstalled-pkgs/paddlepaddle_env_installed_packages.md)
   3. [pytorch](preinstalled-pkgs/pytorch_env_installed_packages.md)
   4. [tensorflow](preinstalled-pkgs/tensorflow_env_installed_packages.md)    (Updated)



## Submitted Files

When the participants submit their developed code and model in a zip file, e.g. your-folder-name.zip, 
the extracted items should look like as follows: 

```
   ./your-folder-name         (rename it acording to your setting)
   | --- __init__.py         
   | --- predict.py           (required) (rename it acording to your setting)
   | --- prepare.py           (required) (DO NOT rename it)
   | --- ... 
   | --- ./your-model-folder  (optional)
   | --- ... 
```

In the extracted folder, 
one prediction-like script (e.g. the predict.py in the baseline code) is required.
In the prediction-like script, the forecast() interface should be implemented, 
and the forecast interface takes a dictionary that consists of a number of settings as the input parameter. 
And, another script named 'prepare.py' is required. 
Note that, the name of this script should be 'prepare.py'.
In the prepare.py script, the prep_env() interface is required to be implemented which will be invoked 
by the evaluation.py script (see the evaluation.py for more details). 

(UPDATE) ATTENTION: The argument parser is not allowed in the prepare.py (please refer to the newly updated tests/test-1.zip), 
since it will conflict with the AIStudio eval_main() system call. 

In particular, the following arguments are required to be declared in 'prepare.py' for evaluation: 
   * checkpoints
   * pred_file
   * start_col
   * framework


## Backend Files for Evaluation

Here, shown as below, we additionally demonstrate how the files are organized in the backend on the server side. 
This folder contains four evaluation sub-folders (i.e. './base', './paddlepaddle', './pytorch' and './tensorflow') and other related sub-folders. 
In each evaluation sub-folder, evaluation.py, metrics.py and test_data.py are placed, which are the same as currently released version 
(please check for the latest updates in the [github](https://github.com/PaddlePaddle/PaddleSpatial/tree/main/apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation/paddlepaddle)).
Besides, apt-requirements.txt and pip-requirements.txt are placed if necessary for different environments. 
If someone finds the environment (see [here](./preinstalled-pkgs) for detailed specifications) does not support 
your model, please provide the detailed (apt/pip)-requirements.txt and contact the organizers ASAP. 

```
   ./some-folder-name
   | --- __init__.py         
   | --- ./base
      | --- __init__.py
      | --- evaluation.py
      | --- metrics.py
      | --- test_data.py
      | --- apt-requiretments.txt   (optional)
      | --- pip-requirements.txt    (optional)
      | --- ./tests
            | --- test-1.zip
            | --- test-2.zip
            | --- ...
   | --- ./paddlepaddle
      | --- __init__.py
      | --- evaluation.py
      | --- metrics.py
      | --- test_data.py
      | --- apt-requiretments.txt   (optional)
      | --- pip-requirements.txt    (optional)
      | --- ./tests
            | --- test-1.zip
            | --- test-2.zip
            | --- ...
   | --- ./pytorch
      | --- __init__.py
      | --- evaluation.py
      | --- metrics.py
      | --- test_data.py
      | --- apt-requiretments.txt   (optional)
      | --- pip-requirements.txt    (optional)
      | --- ./tests
            | --- test-1.zip
            | --- test-2.zip
            | --- ...
   | --- ./tensorflow
      | --- __init__.py
      | --- evaluation.py
      | --- metrics.py
      | --- test_data.py
      | --- apt-requiretments.txt   (optional)
      | --- pip-requirements.txt    (optional)
      | --- ./tests
            | --- test-1.zip
            | --- test-2.zip
            | --- ...
   | --- ...
```


