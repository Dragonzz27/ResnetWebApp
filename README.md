## ResnetWebApp

> 2024/1/10

[![pF9P7FJ.png](https://s11.ax1x.com/2024/01/10/pF9P7FJ.png)](https://imgse.com/i/pF9P7FJ)

### Example shown

choose a picture to upload from local and submit.
[![pF9i7B8.png](https://s11.ax1x.com/2024/01/10/pF9i7B8.png)](https://imgse.com/i/pF9i7B8)

the picture you upload will be shown and prediction result will also be given.
[![pF9iqAg.png](https://s11.ax1x.com/2024/01/10/pF9iqAg.png)](https://imgse.com/i/pF9iqAg)

### Use Docker to run the project(recommended)

1. build an image using Dockerfile
```shell
docker build -t resnetwebapp:latest .
```

2. run an container to start web service
```shell
docker run -p 5000:5000 resnetwebapp
```

3. website url (change the last line in Dockerfile to deploy on servers)

```shell
127.0.0.1:5000
```

### Use Anaconda to run the project

1. create conda environment

```shell
conda create --name ResnetWebApp python=3.10
conda activate ResnetWebApp
```

2. install cuda and pytorch based on your configuration

using **nvidia-smi** to find out version of cuda

```shell
nvidia-smi
```

Take my computer as an example

```shell
Wed Jan 10 13:58:23 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 537.13                 Driver Version: 537.13       CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                     TCC/WDDM  | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3060 ...  WDDM  | 00000000:01:00.0 Off |                  N/A |
| N/A   53C    P8              10W /  95W |    303MiB /  6144MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A     10248    C+G   ...n\120.0.2210.121\msedgewebview2.exe    N/A      |
|    0   N/A  N/A     11056    C+G   ...\Docker\frontend\Docker Desktop.exe    N/A      |
|    0   N/A  N/A     11632    C+G   ...nt.CBS_cw5n1h2txyewy\SearchHost.exe    N/A      |
|    0   N/A  N/A     11660    C+G   ...2txyewy\StartMenuExperienceHost.exe    N/A      |
|    0   N/A  N/A     14656    C+G   ...CBS_cw5n1h2txyewy\TextInputHost.exe    N/A      |
|    0   N/A  N/A     15120    C+G   ...B\system_tray\lghub_system_tray.exe    N/A      |
|    0   N/A  N/A     15628    C+G   ...cal\Microsoft\OneDrive\OneDrive.exe    N/A      |
|    0   N/A  N/A     16160    C+G   ...12.0_x64__8wekyb3d8bbwe\GameBar.exe    N/A      |
|    0   N/A  N/A     16932    C+G   ...m Files\TencentDocs\TencentDocs.exe    N/A      |
|    0   N/A  N/A     19720    C+G   ...__8wekyb3d8bbwe\WindowsTerminal.exe    N/A      |
|    0   N/A  N/A     21804    C+G   ...Programs\Microsoft VS Code\Code.exe    N/A      |
+---------------------------------------------------------------------------------------+
```

From the result we can find the cuda's version is **12.2**, so just install pytorch satisfied with cuda

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. Check the installation of pytorch and install other requirements
    
Just use ipython integrated terminal or something others to check out

```shell
python

In [1]: import torch

In [2]: torch.cuda.is_available()
Out[2]: True
```

Install other packages to run the code

```shell
pip install -r requirement.txt
```

4. Then you can run the code

run **/train/train.ipynb** to adjust code and train resnet18 model given by pytorchvision.

run **/train/eval.ipynb** to evaluate the trained model and compare with others.

run **/web/app.py** to launch a website where you can upload picutres and get an result of classification.

**Notice: you should run /web/app.py under the folder /web, these some bugs with this**

###  Web url for the data
[Datasets On Kaggle](https://www.kaggle.com/datasets/sharumaan/semimages-299x299)

### Where does the idea come from
I watch the [video](https://www.bilibili.com/video/BV1Tb4y1j7iY/), and it interests me.
So, I think that it is meaningful to realize the project by myself.

### What I do in the process
I fully realize the project according to the video, and deploy as container (docker) on cpu-only machine.
I modify some code to satisfy the need since it is hard for container to use a gpu.
So, finally I can train the model on my own machine using a Nvidia GTX 3060, and depoly on my Cloud Server which is running without an gpu.