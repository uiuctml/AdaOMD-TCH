# Adaptive Online Mirror Descent for Tchebycheff Scalarization in Multi-Objective Learning

This is the official repository for the paper [Adaptive Online Mirror Descent for Tchebycheff Scalarization in Multi-Objective Learning](https://arxiv.org/abs/2410.21764).

The synthetic and federated learning (FL) experiments can be reproduced with the corresponding folders. Please follow the ReadMe file in each folder for specific instructions. For quick access to the implementations of our methods (Ada)OMD-TCH, check the following scripts:

| Method | File |
| - | - |
| OMDgd-TCH | `FL/scripts/methods/AFL.py` |
| AdaOMDgd-TCH | `FL/scripts/methods/AFL_new.py` |
| OMDeg-TCH | `FL/scripts/methods/AFLeg.py` |
| AdaOMDeg-TCH | `FL/scripts/methods/AFLeg_new.py` |

### Citation
```bibtex
@article{liu2024online,
  title={Online mirror descent for tchebycheff scalarization in multi-objective optimization},
  author={Liu, Meitong and Zhang, Xiaoyuan and Xie, Chulin and Donahue, Kate and Zhao, Han},
  journal={arXiv preprint arXiv:2410.21764},
  year={2024}
}
```