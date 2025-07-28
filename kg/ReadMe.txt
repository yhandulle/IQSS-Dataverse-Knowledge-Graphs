### ReadMe.txt ###

This is my usual environment.

0.0 install conda on your fas-rc account
========================================
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# and allow for conda init 

1.0 create an environment with conda NOT mamba as follows
=========================================================

conda create -n cuda python=3.10

conda activate cuda
pip3 install accelerate peft bitsandbytes transformers trl
pip install huggingface-hub wandb

2.0 go to huggingface an create a write token, save it or you will lose it
==========================================================================

3.0 go back to fasrc cluster and do 
===================================

huggingface-cli login     [ paste your write token]

4.0: to work with a gpu, the minimal set up is as follows
=========================================================

4.1: allocate a gpu for interactive work
    salloc -p gpu_test -N 1 -n 4 --gres=gpu:1 -t 2:0:0  --mem=21g

4.2: load all the necessary toolkits    
    module load nvhpc/23.7-fasrc01
    module load cuda/12.2.0-fasrc01 
    module load gcc/12.2.0-fasrc01

4.3: make sure you now activate the cuda environment    
    conda activate cuda

In summary copy & paste these commands:
    salloc -p gpu_test -N 1 -n 4 --gres=gpu:1 -t 2:0:0  --mem=21g
    module load nvhpc/23.7-fasrc01
    module load cuda/12.2.0-fasrc01 
    module load gcc/12.2.0-fasrc01
    conda activate cuda

5.0 build llama.cpp
===================

5.1 clone llama.cpp locally with
    git clone https://github.com/ggerganov/llama.cpp.git

5.2 build it as follows: (might take hours)
    cd llama.cpp
    cmake -B build -DGGML_CUDA=ON  # without this you won't get acceleration
    cmake --build build --config Release

6.0 Install llama-cpp-python
============================
This is always a pain, especially if you do not follow the guidelines above
Try this:

    salloc -p gpu_test -N 1 -n 4 --gres=gpu:1 -t 2:0:0  --mem=21g
    module load nvhpc/23.7-fasrc01
    module load cuda/12.2.0-fasrc01 
    module load gcc/12.2.0-fasrc01
    conda activate cuda

    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

cross your fingers, it can takes a lot of time and still fail

If this works, you can run my script.

An alternative quick way is to get rid of the CUDA and just pip install llama-cpp-python.
You'll get the cpu versionm inference will be much slower but still working. Hopefully.

