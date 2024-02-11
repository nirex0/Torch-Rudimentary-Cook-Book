### Create an environment

```
conda create --name torch_env python=3.10
conda deactivate
conda activate torch_env
pip install torch torchvision torchaudio
```

With the ```torch_env``` environment shell active, run these commands:

```
cd vae
python autoencoder.py
python gaussian.py
```