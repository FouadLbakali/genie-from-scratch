# genie-from-scratch

PyTorch implementation of [Genie](https://arxiv.org/abs/2402.15391) (personal project, work in progress).

## Status

Only the video tokenizer (spatio-temporal VQ-VAE) is implemented so far.

- [stvivit.py](stvivit.py) — ST-Transformer encoder/decoder (spatio-temporal ViT)
- [vqvae.py](vqvae.py) — vector quantization

## TODO

- [x] Video tokenizer (ST-ViViT + VQ-VAE)
- [x] Create Dataset
- [ ] Training tokenizer
- [ ] Latent Action Model
- [ ] Dynamics Model (MaskGIT-style or LeWorldModel-style)
- [ ] Training latent action and dynamics model
- [ ] Inference / interactive generation

## Dataset

Training is planned on [fouadlbakali/coinrun-test](https://huggingface.co/datasets/fouadlbakali/coinrun-test).

## References

- Bruce, J. et al. *Genie: Generative Interactive Environments.* arXiv:2402.15391, 2024.
- van den Oord, A., Vinyals, O., Kavukcuoglu, K. *Neural Discrete Representation Learning.* NeurIPS, 2017. arXiv:1711.00937.
- Dosovitskiy, A. et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR, 2021. arXiv:2010.11929.
- Xu, M. et al. *Spatial-Temporal Transformer Networks for Traffic Flow Forecasting.* arXiv:2001.02908, 2020.
- Cobbe, K., Klimov, O., Hesse, C., Kim, T., Schulman, J. *Quantifying Generalization in Reinforcement Learning.* ICML, 2019. arXiv:1812.02341.
