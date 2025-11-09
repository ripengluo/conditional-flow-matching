# conditional-flow-matching-rluo

> Modified from [TorchCFM](https://github.com/atong01/conditional-flow-matching)  
> This fork adapts the flow-matching framework for **crystal-structure generation tasks**, including joint generation of coefficients and lattice degrees of freedom, and modifies the U-Net backbone accordingly.

## Major Changes from Upstream
- Sampling pipeline (`conditional_flow_matching.py`, `optimal_transport.py`) has been extended to crystal-structure domain:  
  - Now supports loading crystal structure data (CIF files, etc.) and generating tasks based on lattice and coefficient variables.  
  - Implements **joint generation**: not just one variable, but multiple correlated variables (e.g., coeff + lattice DOF) in one model.  
- Network architecture (`unet`):  
  - Added/modified embedding layer to accept crystal structural inputs (e.g., lattice dof, space group, atomic types).  
  - Modified output head to output multiple targets (coefficients + lattice DOF).  
  - Updated UNet blocks: [e.g., number of channels changed from 64→128, additional skip connections for structural features].  
- Usage/Environment:  
  - Requires Python 3.10+, PyTorch 2.9.0+, CUDA 12.9, etc.  
  - Large model weights (>100 MB) are tracked via Git LFS.  
- Licensing & Acknowledgements:  
  - Original work by TorchCFM (MIT Licence). This fork retains original copyright and adds modifications © 2025 Ripeng Luo.  
  - Please cite original work and this fork when using.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

