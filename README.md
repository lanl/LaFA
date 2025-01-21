# **LaFA: Latent Feature Attacks on Non-negative Matrix Factorization**

This repository contains the official implementation of **LaFA** (Latent Feature Attacks), as introduced in the paper:  
> [**Latent Feature Attacks on Non-negative Matrix Factorization**](https://arxiv.org/abs/2408.03909)  
> *Authors: Minh Vu, Ben Nebgen, Erik Skau, Geigh Zollicoffer, Juan Castorena, Kim Rasmussen, Boian Alexandrov, Manish Bhattarai 

---

## **Abstract**

As Machine Learning (ML) applications rapidly grow, concerns about adversarial attacks compromising their reliability have gained significant attention. One unsupervised ML method known for its resilience to such attacks is Non-negative Matrix Factorization (NMF), an algorithm that decomposes input data into lower-dimensional latent features. However, the introduction of powerful computational tools such as Pytorch enables the computation of gradients of the latent features with respect to the original data, raising concerns about NMF's reliability. Interestingly, naively deriving the adversarial loss for NMF as in the case of ML would result in the reconstruction loss, which can be shown theoretically to be an ineffective attacking objective. In this work, we introduce a novel class of attacks in NMF termed Latent Feature Attacks (LaFA), which aim to manipulate the latent features produced by the NMF process. Our method utilizes the Feature Error (FE) loss directly on the latent features. By employing FE loss, we generate perturbations in the original data that significantly affect the extracted latent features, revealing vulnerabilities akin to those found in other ML techniques. To handle large peak-memory overhead from gradient back-propagation in FE attacks, we develop a method based on implicit differentiation which enables their scaling to larger datasets. We validate NMF vulnerabilities and FE attacks effectiveness through extensive experiments on synthetic and real-world data. 

For details, refer to the full paper: [arXiv:2408.03909](https://arxiv.org/abs/2408.03909).

---

## **Features**

- **Latent Feature Attacks (LaFA)**:
  - Directly targets latent features extracted by NMF.
  - Employs **Feature Error (FE) Loss** to craft adversarial perturbations.
- **Implicit Differentiation**:
  - Scales FE attacks to larger datasets by reducing memory overhead.
- **Comprehensive Experiments**:
  - Validates attack effectiveness on datasets like WTSI, MNIST, and synthetic data.
- **Visualization Tools**:
  - Includes scripts to analyze attack impacts and evaluate performance.

---

## **Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/your_username/LaFA.git
cd LaFA
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
pip install -e .
```

---

## **Usage**

### **1. Running Experiments**
Below are some examples of how to run experiments using **LaFA**:

#### **Example 1: Basic Configuration**
Perform **iterative** attacks on the Face dataset using the **L2 norm** without implicit differentiation:
```bash
python main.py --dataset Face --rank 5                --base_iter 10000 --nmf_iter 2000                --iterative                --no_iter 40 --taylor 100 --norm L2                --eps_min 0.0 --eps_max 0.02 --alpha 0.01 --no_eps 20                --seed 2711
```

#### **Example 2: Implicit Differentiation with Iterative Attack**
Enable **implicit differentiation** for memory-efficient gradient computation with the **L2 norm**:
```bash
python main.py --dataset Face --rank 5                --base_iter 10000 --nmf_iter 2000                --implicit                --iterative                --no_iter 40 --taylor 100 --norm L2                --eps_min 0.0 --eps_max 0.02 --alpha 0.01 --no_eps 20                --seed 2711
```

#### **Example 3: Linf Norm Attack with Implicit Differentiation**
Perform an **Linf norm** attack using implicit differentiation to target a stricter adversarial region:
```bash
python main.py --dataset Face --rank 5                --base_iter 10000 --nmf_iter 2000                --implicit                --iterative                --no_iter 40 --taylor 100 --norm Linf                --eps_min 0.0 --eps_max 0.01 --alpha 0.01 --no_eps 20                --seed 2711
```

#### **Example 4: Iterative Linf Norm Attack Without Implicit Differentiation**
Run an **iterative attack** using the **Linf norm**, without implicit differentiation:
```bash
python main.py --dataset Face --rank 5                --base_iter 10000 --nmf_iter 2000                --iterative                --no_iter 40 --taylor 100 --norm Linf                --eps_min 0.0 --eps_max 0.01 --alpha 0.01 --no_eps 20                --seed 2711
```

---

### **Batch Job Scripts**

To streamline the execution of multiple experiments, you can leverage the **Bash scripts** located in the `sbatch_jobs` directory. These scripts allow you to run experiments in batch mode for various datasets and configurations, enabling efficient reproduction of the results presented in the paper.

---

### **Available Scripts**

| **Script Name**           | **Description**                                                                          |
|----------------------------|------------------------------------------------------------------------------------------|
| `submit_jobs_face.sh`      | Batch job script for running experiments on the **Face dataset**.                        |
| `submit_jobs_mnist.sh`     | Batch job script for running experiments on the **MNIST dataset**.                       |
| `submit_jobs_synthetic.sh` | Batch job script for running experiments on **synthetic datasets**.                      |
| `submit_jobs_wtsi.sh`      | Batch job script for running experiments on the **WTSI dataset**.                        |

---

### **How to Use Batch Scripts**

Each script contains pre-defined configurations for the respective datasets and can be executed using the following command:

```bash
sbatch sbatch_jobs/submit_jobs_<dataset>.sh

```

## **Citing This Work**

If you use this code or work in your research, please cite:
```
@article{vu2024lafa,
  title={LaFA: Latent Feature Attacks on Non-negative Matrix Factorization},
  author={Vu, Minh and Nebgen, Ben and Skau, Erik and Zollicoffer, Geigh and Castorena, Juan and Rasmussen, Kim and Alexandrov, Boian and Bhattarai, Manish},
  journal={arXiv preprint arXiv:2408.03909},
  year={2024}
}

```

---

## **License**

## Copyright Notice
>© 2025. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.

**LANL O Number: O4797**

## License
This program is open source under the BSD-3 License.
Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.
 
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution.
 
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

