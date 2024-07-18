# DGDM : Deterministic Guidance Diffusion Model for Probabilistic Weather Forecasting

Welcome to the official repository for the **[Deterministic Guidance Diffusion Model for Probabilistic Weather Forecasting](https://arxiv.org/abs/2312.02819)**

<img src="resources/architecture.png">  

## Preparations 

Before diving into the model, ensure your environment is set up correctly.

### Setup
```bash
conda create -n [name] python==3.9
conda activate [name]
pip install -r requirements.txt
```

### Dataset
#### Moving MNIST dataset
```bash
cd ./data/moving_mnist
bash download_mmnist.sh
```

#### PWN-Typhoon
- Download PWN-Typhoon dataset [Link](https://github.com/xuekt98/BBDM) and set ```data``` dir.

# Usage
## Demo
TODO

## Train

```bash
python main.py -c configs/Template-MovingMNIST.yaml -t -r set/your/save/dir
```

## Test

```bash
python main.py -c configs/Template-MovingMNIST.yaml -r set/your/save/dir
```

## Citation
```latex
@misc{yoon2023deterministic,
      title={Deterministic Guidance Diffusion Model for Probabilistic Weather Forecasting}, 
      author={Donggeun Yoon and Minseok Seo and Doyi Kim and Yeji Choi and Donghyeon Cho},
      year={2023},
      eprint={2312.02819},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Reference
This repository was implemented by the following repositories.
- [BBDM](https://github.com/xuekt98/BBDM)
- [VDM](https://github.com/lucidrains/video-diffusion-pytorch)
- [SimVP](https://github.com/A4Bio/SimVP-Simpler-yet-Better-Video-Prediction)
