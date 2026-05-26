# Deepfake detector based on micro-expression analysis for online conferences 

This deepfake detector consists of three detection models: a trained ResNet50+LSTM on full face frames, a trained ResNet50+LSTM on frames with eyes, and an [LRNet](https://github.com/frederickszk/LRNet) model trained on face landmarks.


## Content
- [Quick start](#-quickstart)
- [Install requirements](#-requirements)
- [Configuring paths](#-configuring)
- [Run detector](#-running)
- [Files structure](#-structure)


---

## Quick start
```bash
git clone [URL_репозитория]
python -m venv venv
source venv/bin/activate
```

## Requirements
```bash
pip install -r requirements.txt
```
python main.py
