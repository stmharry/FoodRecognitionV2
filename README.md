# SOP for Building Residual Network-50 for Food Classification

0. Do not plug GPU
0. Install `Ubuntu 16.04 LTS`
(If upgraded from 14.04, see [here](https://www.digitalocean.com/community/tutorials/how-to-upgrade-to-ubuntu-16-04-lts))
0. Install `nvidia-367` graphics driver, and then plug GPU back

    ```bash
    sudo apt-add-repository ppa:graphics-drivers/ppa
    sudo apt-get update
    sudo apt-get install nvidia-367
    ```
0. Install `CUDA 8.0` without installing graphics driver (download [here](https://developer.nvidia.com/cuda-toolkit), use `.run` to optionally disable installing the graphics driver), and install patch for `gcc 5.4`

    ```bash
    sudo sh cuda_8.0.27_linux.run --silent --toolkit --override
    sudo sh cuda_8.0.27.1_linux.run --silent --accept-eula
    ```
0. Install `cuDNN 5.1.5` (download [here](https://developer.nvidia.com/cudnn))

    ```bash
    sudo tar -zxvf cudnn-8.0-linux-x64-v5.1.tgz --directory /usr/local/
    ```
0. `apt-get` dependencies

    ```bash
    sudo apt-get install python-pip python-dev python-wheel python-numpy git zlib1g-dev swig imagemagick
    ``` 
    
0. `pip` dependencies

    ```bash
    pip install scipy
    ```  
0. Install `JDK 8`

    ```bash
    sudo add-apt-repository ppa:webupd8team/java
    sudo apt-get update
    sudo apt-get install oracle-java8-installer
    ```
0. Install Bazel (follow [here](http://www.bazel.io/docs/install.html))
    
    ```bash
    wget https://github.com/bazelbuild/bazel/releases/download/0.3.0/bazel-0.3.0-installer-linux-x86_64.sh
    chmod +x bazel-0.3.0-installer-linux-x86_64.sh
    ./bazel-0.3.0-installer-linux-x86_64.sh --user
    ```
0. Install TensorFlow from source (follow [here](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#installing-from-sources))

    ```bash 
    git clone https://github.com/tensorflow/tensorflow 
    cd tensorflow
    git checkout ea9e00a630f91a459dd5858cb22e8cd1a666ba4e
    git pull
    ./configure [CUDA: 8, cuDNN: 5.1.5, compute capability: 6.1]
    bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
    pip install /tmp/tensorflow_pkg/tensorflow-* [the name depends]
    ```

0. Clone project repository

    ```bash
    git clone http://dev.2bite.com:14711/stmharry/ResidualNetwork.git
    ```
