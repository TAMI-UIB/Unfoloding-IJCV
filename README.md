# [IJCV 2025] Nonlocal Retinex-Based Variational Model and its Deep Unfolding Twin for Low-Light Image Enhancement

## [Paper](https://link.springer.com/article/10.1007/s11263-025-02551-y)


This repository contains the implementation and additional resources for the following paper:

**Nonlocal Retinex-Based Variational Model and its Deep Unfolding Twin for Low-Light Image Enhancement**  
*Daniel Torres, Joan Duran, Julia Navarro, Catalina Sbert*  
<!--Submmited to the International Journal of Computer Vision-->

---

## 📄 Abstract
Images captured under low-light conditions present significant limitations in many applications, as poor lighting can obscure details, reduce contrast, and hide noise. Removing the illumination effects and enhancing the quality of such images is crucial for many tasks, such as image segmentation and object detection. In this paper, we propose a variational method for low-light image enhancement based on the Retinex decomposition into illumination, reflectance, and noise components. A color correction pre-processing step is applied to the low-light image, which is then used as the observed input in the decomposition. Moreover, our model integrates a novel nonlocal gradient-type fidelity term designed to preserve structural details. Additionally, we propose an automatic gamma correction module. Building on the proposed variational approach, we extend the model by introducing its deep unfolding counterpart, in which the proximal operators are replaced with learnable networks. We propose cross-attention mechanisms to capture long-range dependencies in both the nonlocal prior of the reflectance and the nonlocal gradient-based constraint. Experimental results demonstrate that both methods compare favorably with several recent and state-of-the-art techniques across different datasets. In particular, despite not relying on learning strategies, the variational model outperforms most deep learning approaches both visually and in terms of quality metrics.

---

## 🛠️ Environment

1. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
---

## ⚙️ Setup

You should list your dataset as the following structure:

<pre> dataset/ 
         your_dataset/ 
            train/ 
               high/ 
               low/ 
            eval/ 
               high/
               low/ </pre>

---
## Train

Run the following command:
   ```bash
   python train.py 
   ```
---
## Test

Run the following command:

   ```bash
   python test.py
   ```

---
## 📌 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{torres2025nonlocal,
  title={Nonlocal Retinex-Based Variational Model and its Deep Unfolding Twin for Low-Light Image Enhancement},
  author={Torres, Daniel and Duran, Joan and Navarro, Julia and Sbert, Catalina},
  journal={International Journal of Computer Vision},
  pages={1--22},
  year={2025},
  publisher={Springer}
}
```

---
## Acknowledgements

This work was funded by MCIN/AEI/10.13039/501100011033/ and by the European Union NextGenerationEU/PRTR via the MaLiSat project TED2021-132644B-I00.
