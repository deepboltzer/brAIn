# Fixes for some issues regarding gym and pytorch

## Missing .dll (ale_c.dll)

$\Rightarrow$ [Fix for module could not be found](https://github.com/openai/gym/issues/1726)

## Create a virtual environment with anaconda

Note new virtual environments sometimes (?) don't have `pip` installed per default:

```
conda create --name NAME
conda activate NAME
conda install pip
```

## Install correct PyTorch (cuda 10.2)

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

If you run into problems this might be helpful [StackOverflow on torch.cuda.is_available() returns False](https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with)

## Install gym atari roms

Note that roms are no longer included with the gym installation, therefore they'll have to be installed separately using AutoROM:

```
pip install autorom
AutoROM
ale-import-roms /PATH/TO/ENV/lib/site-packages/AutoROM/roms
```

Test if gym is installed correctly:

```
python
>>>import gym
gym.make('ALE/Breakout-v5')
```

This should not throw any errors.
