# ml-agents-visual-observations-docker-GPU
A sample Docker image for using ML-Agents with GPU support for Unity. 

In my other repository (https://github.com/ValerioB88/ml-agents-visual-observations-docker) I showed how to use ml-agents within a Docker container with visual observations by running a virtual frame buffer. The drawback of that approach is that Unity rendering is not exploiting the hardware capability (namely, your powerful GPU). The GPU will normally runs your neural network's computation, but if your NN has to wait for Unity to generate new frames, then the whole training time is gonna suffer. 
You can use this Docker image and the instructions in this repo to overcomes the issue. 

# Usage
### Prerequisites
* **NVIDIA Drivers**: On the host machine, make sure that the NVIDIA drivers are installed. To check that, run `nvidia-smi`, and something like this should appear: 
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 455.38       Driver Version: 455.38       CUDA Version: 11.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  TITAN X (Pascal)    Off  | 00000000:01:00.0 Off |                  N/A |
| 23%   36C    P8    16W / 250W |     52MiB / 12192MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A       971      G   /usr/lib/xorg/Xorg                 49MiB |
+-----------------------------------------------------------------------------+
```
If it doesn't, install the NVIDIA drivers (plenty of online tutorials about that, but it _should_ just be something like `sudo ubuntu-drivers autoinstall` or `sudo apt install nvidia-driver-<version-number>` e.g. `sudo apt install nvidia-driver-430`).

* **NVIDIA Container Toolkit**: Check if you have it by running `sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`. If you don't, follow these instructionts  https://www.server-world.info/en/note?os=Ubuntu_20.04&p=nvidia&f=2

* **Set up the xorg.config file**: follow these instructions: https://virtualgl.org/Documentation/HeadlessNV. Don't worry about the VirtualGL part, but follow the points 1, 2 and 3.

To check that everything is setup correctly on the host machine, open two terminals. On one of them run `nvidia-smi -lms 100`, on the other run `export DISPLAY=:0 && glxgears -display: 0`. `glxgears` it's just a little OpenGL demo that you can easily use as a test. You should see on the `nvidia` terminal the GPU crunching numbers to render the glxgear script (if you don't have glxgears, install it with `apt update && apt install mesa-utils`). You should get something like this:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 455.38       Driver Version: 455.38       CUDA Version: 11.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  TITAN X (Pascal)    Off  | 00000000:01:00.0 Off |                  N/A |
| 23%   39C    P0    81W / 250W |     57MiB / 12192MiB |     81%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A       971      G   /usr/lib/xorg/Xorg                 51MiB |
|    0   N/A  N/A     12582      G   glxgears                            3MiB |
+-----------------------------------------------------------------------------+
```
## Run it 
Build the Docker image with `docker build --tag ml-agents-visobs-gpu .`. Once done, run it with
```bash
docker run -it \
    -e DISPLAY=:0 \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --runtime=nvidia \
    --entrypoint /bin/bash \
    ml-agents-visobs-gpu
```
Once inside the terminal, you can test it with glxgears again. (You can use `nvidia-smi` on the host machine.) If it works, we are ready to run a Unity scene.
To test whether Unity in ml-agents is actually running on the GPU, use this script in `python` inside the Docker container:
```python
import numpy as np
import time
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.registry import default_registry

env = default_registry['VisualHallway'].make()
env.reset()
behaviour_names = list(env.behavior_specs.keys())
decision_steps, terminal_steps = env.get_steps(behaviour_names[0])
behavior_name = list(env.behavior_specs)[0] 
spec = env.behavior_specs[behavior_name]

a=time.time()
for i in range(100):
	action = spec.create_random_action(len(decision_steps)); env.set_actions(behavior_name, action);env.step();

print(time.time() - a)
```

If it runs without errors, you are good to go! You should see the GPU running the Unity process and crunching numbers on your `nvidia-smi` terminal. I got a speed-up of 2x with this configuration, compared to the `xvfb` trick I was using before. This is _not great_, but **better**. However, I can at least assume that the bottleneck now is not due to Unity3D rendering: it may be due to socket transfer, or to the rest of my Unity scene that is not 3D related. 
Take this image and mix it up with your own. Enjoy!
