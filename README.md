# Amplified-MIA - Official Pytorch Implementation

Official Pytorch implementation for paper:

Z. Zhang, X. Wang, J. Huang and S. Zhang, "Analysis and Utilization of Hidden Information in Model Inversion Attacks," in IEEE Transactions on Information Forensics and Security, doi: [10.1109/TIFS.2023.3295942](https://doi.org/10.1109/TIFS.2023.3295942).

## Abstract

The widely applications of deep learning have raised concerns about the privacy issues in deep neural networks. Model inversion attack aims to reconstruct specific details of each private training sample from a given neural network. However, limited to the availability of useful information, reconstructing distinctive private training samples still has a long way to go. In this paper, the requirements to reconstruct distinctive private training samples are investigated using information entropy. We find that more information is needed to reconstruct distinctive samples and propose to use the often ignored hidden information to achieve this goal. To better utilize this information, Amplified-MIA is proposed. In Amplified-MIA, a nonlinear amplification layer is inserted between the target network and the attack network. This nonlinear amplification layer further contains a nonlinear amplification function. The definition of the nonlinear amplification function is given and the effect of this nonlinear amplification function on the entropy of the hidden information is derived. The proposed nonlinear amplification function can amplify the small prediction vector entries and enlarge the differences between different prediction vectors in the same class. Thus, the hidden information can be better utilized by the attack network and distinctive private samples can be reconstructed. Various experiments are performed to empirically analyze the effects of the nonlinear amplification function on the reconstruction results. The reconstruction results on three different datasets show that the proposed Amplified-MIA outperforms existing works on almost all tasks. Especially, it achieves up to 68% performance gain of the Pixel Accuracy score over the direct inversion method on the hardest face reconstruction task.

## Usage

### Required Runtime Libraries

* [Anaconda](https://www.anaconda.com/download/)
* [Pytorch](https://pytorch.org/) -- `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
* [zhangzp9970/torchplus](https://github.com/zhangzp9970/torchplus) -- `conda install torchplus -c zhangzp9970`

The code is compatable with the latest version of all the software.

### File Description

* main_MNIST.py -- train the MNIST classifier.
* main_FaceScrub.py -- train the FaceScrub classifier.
* attack_MNIST.py -- perform Amplified-MIA attack against the trained MNIST classifier before.
* attack_FaceScrub.py -- perform Amplified-MIA attack against the trained FaceScrub classifier before.
