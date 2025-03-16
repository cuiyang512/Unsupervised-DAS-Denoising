# Unsupervised  DAS Denoising
### Unsupervised Deep Learning for DAS-VSP Denoising Using Attention-Based Deep Image Prior

Distributed Acoustic Sensing (DAS) has emerged as a widely used technology in various applications, including borehole microseismic monitoring, active source exploration, and ambient noise tomography. Compared with conventional geophones, a fiber optic cable has unique characteristics that allow it to withstand high-temperature and high-pressure environments. However, due to its high sensitivity, the obtained seismic records are often corrupted with unavoidable background noise, which introduces more uncertainty in the subsequent seismic data processing and interpretation. Thus, the development of robust denoising techniques for DAS data is crucial to minimize the impact of noise and enhance the reliability of seismic data processing and interpretation. In this work, we propose a ground-truth-free method for strong background noise suppression in Distributed Acoustic Sensing Vertical Seismic Profiling (DAS-VSP) data. Compared to existing deep learning methods, the proposed approach demonstrates promising generalizability in handling field examples across different surveys. The proposed method consists of four stages: training set extension with a patching scheme, feature selection with a kurtosis-based method, denoising with a deep image prior (DIP)-based unsupervised neural network, and an unpatching approach for denoised data reconstruction. Numerical experiments conducted on synthetic data and several profiles from the Utah FORGE project and the Groß Sch¨onebeck site demonstrate that the proposed method can effectively suppress most of the background noise while preserving hidden signals. Furthermore, the unsupervised learning approach is unconditionally generalizable when applied to vastly different field data because it does not require pre-labeled datasets for training.

![Network architecture](Figs/architecture.png)

## Results comparsion on different surveys

### Utah Forge and Groß Schönebeck Examples
<img src="Figs/TGRS_Denoising_example_2_Yang.gif" alt="Forge1" width="200" height="400"> <img src="Figs/TGRS_Denoising_example_3_Yang.gif" alt="Forge2" width="200" height="400"> <img src="Figs/TGRS_Denoising_example_4_Yang.gif" alt="Forge3" width="200" height="400"> <img src="Figs/TGRS_Denoising_example_1_Yang.gif" alt="Groß Schönebeck" width="200" height="400">

## Reference
    Cui, Y., Waheed, U. B., & Chen, Y. (2025). Unsupervised Deep Learning for DAS-VSP Denoising Using Attention-Based Deep Image Prior. IEEE Transactions on Geoscience and Remote Sensing.
    Cui, Yang, Umair bin Waheed, and Yangkang Chen. "Background noise suppression for DAS-VSP data using attention-based deep image prior." SEG International Exposition and Annual Meeting. SEG, 2024.

BibTex

    @article{cui2025unsupervised,
      title={Unsupervised Deep Learning for DAS-VSP Denoising Using Attention-Based Deep Image Prior},
      author={Cui, Yang and Waheed, Umair Bin and Chen, Yangkang},
      journal={IEEE Transactions on Geoscience and Remote Sensing},
      year={2025},
      publisher={IEEE}
    }
    @inproceedings{cui2024background,
      title={Background noise suppression for DAS-VSP data using attention-based deep image prior},
      author={Cui, Yang and Waheed, Umair bin and Chen, Yangkang},
      booktitle={SEG International Exposition and Annual Meeting},
      pages={SEG--2024},
      year={2024},
      organization={SEG}
    }

## Install 
For set up the environment and install the dependency packages, please run the following script:
    
    conda create -n DASd python=3.11.7
    conda activate DASd
    conda install ipython notebook
    pip install tensorflow==2.15.0, keras==2.15.0, h5py==3.10.0, scikit-learn==1.4.0, seaborn==0.13.2, matplotlib==3.8.4, scipy==1.11.4

If you want to run with GPUs, please run the following script to install the dependency package:

    conda create -n DASd python=3.11.7
    conda activate DASd
    conda install ipython notebook
    pip install tensorflow==2.15.0, keras==2.15.0, h5py==3.10.0, scikit-learn==1.4.0, seaborn==0.13.2, matplotlib==3.8.4, scipy==1.11.4
    conda install -c anaconda cudnn==8.2.1

## Development

    The development team welcomes voluntary contributions from any open-source enthusiast. 
    If you want to make contribution to this project, feel free to contact the development team. 
    
## Contact

    Regarding any questions, bugs, developments, collaborations, please contact  
    Yang Cui
    yang.cui512@gmail.com
