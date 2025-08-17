# GateDiffST: Installation Guide  
GateDiffST is a gated diffusion framework for spatial transcriptomics analysis, which adopts a two-stage architecture: first, it denoises gene expression profiles through GCN combined with ZINB loss, then uses a diffusion model incorporating gated linear attention and rotary positional encoding to refine the data, and finally averages the refined data with the denoised data to form the final gene expression profile data.
It mainly addresses issues such as sparsity and inaccurate spatial domain identification in spatial transcriptomics data, improves the accuracy of cell clustering, can better capture long-range spatial dependencies and spatial expression patterns of genes, and provides support for the analysis of tissue microenvironments and other related fields.
This document provides the environment setup and installation steps for GateDiffST.  


## 1. Environment Preparation  
Create a dedicated Conda environment for GateDiffST:  
```bash
# Create and activate the environment
conda create -n GateDiffST
source activate GateDiffST  # For Linux/macOS; use "conda activate GateDiffST" for Windows

# Install Python
conda install python=3.8
```


## 2. Dependencies Installation  
Install required packages via Conda and pip:  

### Conda Packages  
```bash
# Install system-level dependencies
conda install conda-forge::gmp
conda install conda-forge::pot

# Install PyTorch with CUDA support (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Pip Packages  
```bash
pip install scanpy
pip install anndata==0.8.0
pip install pandas==1.4.2
pip install rpy2==3.5.1  # For R package integration
pip install scikit-learn==1.1.1
pip install scipy==1.8.1
pip install tqdm==4.64.0
pip install einops
```


## 3. R Package Installation  
Install the required R package `mclust` via `rpy2`:  
```python
import rpy2.robjects as robjects

robjects.r('''
    install.packages('mclust')
''')
```
## 4. Data Access  
The data used in this study can be accessed through the following channels:  

### 4.1 Supplementary Materials  
Relevant data are included in the Supplementary Materials of our manuscript. Please refer to the corresponding section in the publication for detailed information and download links.  

### 4.2 Google Drive  
You can also directly access the data via Google Drive:  
[https://drive.google.com/drive/folders/1crS8sbX12Qw-jSQd1wzqCZ4qrbPRtRdF](https://drive.google.com/drive/folders/1crS8sbX12Qw-jSQd1wzqCZ4qrbPRtRdF)  

**Note**: Ensure a stable network connection to access the Google Drive link. If you encounter regional access restrictions, we recommend using a VPN service compliant with local laws and regulations, or reach out to the corresponding author for alternative data transfer solutions (e.g., WeTransfer, Baidu Cloud, etc.).  

