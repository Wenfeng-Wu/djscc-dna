# Deep Joint Source-Channel Coding for DNA Image Storage: A Novel Approach With Enhanced Error Resilience and Biological Constraint Optimization

## Authors
Wenfeng Wu; Luping Xiang; Qiang Liu; Kun Yang

## Paper address
https://arxiv.org/pdf/2311.01122

https://ieeexplore.ieee.org/document/10314138

## Abstract

In the current era, DeoxyriboNucleic Acid (DNA) based data storage emerges as an intriguing approach, garnering substantial academic interest and investigation. This paper introduces a novel deep joint source-channel coding (DJSCC) scheme for DNA image storage, designated as DJSCC-DNA. This paradigm distinguishes itself from conventional DNA storage techniques through three key modifications: 1) it employs advanced deep learning methodologies, employing convolutional neural networks for DNA encoding and decoding processes; 2) it seamlessly integrates DNA polymerase chain reaction (PCR) amplification into the network architecture, thereby augmenting data recovery precision; and 3) it restructures the loss function by targeting biological constraints for optimization. The performance of the proposed model is demonstrated via numerical results from specific channel testing, suggesting that it surpasses conventional deep learning methodologies in terms of peak signal-to-noise ratio (PSNR) and structural similarity index (SSIM). Additionally, the model effectively ensures positive constraints on both homopolymer run-length and GC content.

![image](https://github.com/user-attachments/assets/a06bf010-049b-4b4f-b242-142ff88c5759)

## File

DJSCC_DNA.py and DJSCC_DNA_multi.py: The structure of neural network

train.py, train_multi.py and train_nomyloss.py: Neural networks are trained in different ways

train.py, train_multi.py: Neural networks are tested in different ways

functions.py: some fuctions

figures_xxx.py, performance_xxx.py: Drawing codes. In different parameter Settings, test separately and retain data

Other code files: Pictures, functions, program examples


## Citation
@ARTICLE{10314138,
  author={Wu, Wenfeng and Xiang, Luping and Liu, Qiang and Yang, Kun},
  journal={IEEE Transactions on Molecular, Biological, and Multi-Scale Communications}, 
  title={Deep Joint Source-Channel Coding for DNA Image Storage: A Novel Approach With Enhanced Error Resilience and Biological Constraint Optimization}, 
  year={2023},
  volume={9},
  number={4},
  pages={461-471},
  keywords={DNA;Image coding;Encoding;Decoding;Biological information theory;Neural networks;Deep learning;DNA storage;deep learning;joint source-channel coding;biological constraints},
  doi={10.1109/TMBMC.2023.3331579}}
