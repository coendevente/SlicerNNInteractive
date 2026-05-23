# External server setup

Follow the instructions below to set up an nnInteractive computation server application manually and connect to it from 3D Slicer.

Hardware requirements: a Windows or Linux computer with an NVIDIA GPU. 10GB of VRAM is recommended. Small objects should work with <6GB. nnInteractive supports Python 3.10+> (source: [The nnInteractive README](https://github.com/MIC-DKFZ/nnInteractive?tab=readme-ov-file#prerequisites).

After setting up the server, go to the `nnInteractive` module in Slicer, open the `Configuration` tab, select `External server`, and enter the server URL. This should look something like `http://remote_host_name:1527`, or `http://localhost:1527` if running locally. If running the server on the same Windows computer as 3D Slicer, use `localhost` (ignore that the server suggests `0.0.0.0` may be used).

## Server running on Linux

### Option 1: Using Docker

```
docker pull coendevente/nninteractive-slicer-server:latest
docker run --gpus all --rm -it -p 1527:1527 coendevente/nninteractive-slicer-server:latest
```

This will make the server available under port `1527` on your machine. If you would like to use a different port, say `1627`, replace `-p 1527:1527` with `-p 1627:1527`.

### Option 2: Using `uv`

Another option is to run the server with [`uv`](https://docs.astral.sh/uv/) (see `uv` installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
uv run --with nninteractive-slicer-server nninteractive-slicer-server --host 0.0.0.0 --port 1527
```

### Option 3: Using `pip`

_Step 1. Create a Python virtual environment_

If setting up the server for the first time, you need to create a Python virtual environment (e.g., using `conda` or `venv`) by specifying a location on your disk and activating that environment. For example, on Linux, using `venv`, you can accomplish this using these commands:

```bash
python3 -m venv path_to_your_virtual_environment 
source path_to_your_virtual_environment/bin/activate
```

_Step 2. Install the server_

Next, you can install the server to this environment with these commands:

```bash
pip install nninteractive-slicer-server
nninteractive-slicer-server --host 0.0.0.0 --port 1527
```

If you would like to use a different port, say `1627`, replace `--port 1527` with `--port 1627`.

> [!NOTE]  
> Remember that you'll have to start the server again if it was stopped for some reason (e.g., after rebooting your machine). To do so, activate your virtual Python environment with the `source` command above and run `nninteractive-slicer-server --host 0.0.0.0 --port 1527` again to start the server.

> [!NOTE]  
> When starting the server, you can ignore the message `nnUNet_raw is not defined [...] how to set this up.`. Setting up these environment variables is not necessary when using `SlicerNNInteractive`.

## Server running on Windows

### One-time setup

Python and a pytorch package with GPU support is required. You can follow the steps below to set these up on your computer for your user:

1. Download pixi package manager by running this command in `Terminal` (to launch terminal, press the Windows button on your keyboard, type `terminal` and hit `Enter` key):

```
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

2. Close the terminal and open a new Terminal to run the commands below to install Python and pytorch. The last step may take 10 minutes to complete, with no updates on the output for several minutes.

```
cd /d %localappdata%
mkdir nninteractive-server
cd nninteractive-server
pixi init .
pixi add python=3.12 pip
cd .pixi\envs\default\Scripts
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Start the server

To start the server, there is no need to redo the steps above (install pixi and Python), just open `Terminal` and run these commands:

```
cd /d %localappdata%\nninteractive-server\.pixi\envs\default\Scripts
pip install nninteractive-slicer-server
nninteractive-slicer-server --host 0.0.0.0 --port 1527
```

If the firewall asks permission to access the port then allow it.

If you would like to use a different port, say `1627`, replace `--port 1527` with `--port 1627`.

> [!NOTE]  
> When starting the server, you can ignore the message `nnUNet_raw is not defined [...] how to set this up.`. Setting up these environment variables is not necessary when using `SlicerNNInteractive`.
