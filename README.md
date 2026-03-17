# DTFPNet: Temporal and Frequency Dynamic Graph Neural Network for Time Series Classification
@article{yu2025dtfpnet,
  title={DTFPNet: Temporal and Frequency Dynamic Graph Neural Network for Time Series Classification},
  author={Yu, Wuqing and Fu, Bowen and Li, Zhan and Zhou, Jian and Wang, Junhao and Zhang, Jiacai},
  journal={Pattern Recognition},
  pages={112588},
  year={2025},
  publisher={Elsevier}
}

The current codebase retains its original implementation style. This decision was made to preserve the reproducibility of the results presented in this manuscript, as modifications—such as reordering layers in the __init__ function—would impact parameter initialization and consequently hinder the ability to replicate our reported outcomes. <br>

## Datasets
- UCR and UEA classification datasets are available at https://www.timeseriesclassification.com
- Sleep-EDF and UCIHAR datasets are from https://github.com/emadeldeen24/TS-TCC
- For any other dataset, to convert to `.pt` format, follow the preprocessing steps here https://github.com/emadeldeen24/TS-TCC/tree/main/data_preprocessing

## Train and test
- -run the ./DTFPNet_temp/main_run.py
- The test results will be saved in ./textFiles/DTFPNet_UCR.txt
- some test results in this paper can be seen in path ./lightning_logs/    and  ./textFiles/DTFPNet_UCR.txt

## Acknowledgements
The codes in this repository are inspired by the following:
TSLANet: Rethinking Transformers for Time Series Representation Learning https://github.com/emadeldeen24/TSLANet

## 📝 Fix: Typo in IDCT Formula (Eq. 3)

### Description
I noticed a notation error in **Equation 3** (Inverse Discrete Cosine Transform). 

The summation index is currently denoted as $l$ (ranging from $0$ to $L-1$). However, since $f(l)$ is a function of time index $l$, the summation should be performed over the frequency index $\mu$ to match the terms $c(\mu)$ and $F(\mu)$.

### Correction

**Original (Incorrect):**
$f(l) = \sqrt{\frac{2}{L}} \sum_{l=0}^{L-1} c(\mu)F(\mu)\cos[\frac{\pi\mu(2l+1)}{2L}]$

**Corrected:**
$f(l) = \sqrt{\frac{2}{L}} \sum_{\mu=0}^{L-1} c(\mu)F(\mu)\cos[\frac{\pi\mu(2l+1)}{2L}]$

### Changes
- Changed summation index from $l$=0 to $\mu$=0 in the IDCT formula.

Contact email: 202431081038@mail.bnu.edu.cn
