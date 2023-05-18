# Model-driven analysis of ECG using reinforcement learning

This repository contains the code used for the paper entitled "Model-driven analysis of ECG using reinforcement learning" archived at [preprints.org](https://www.preprints.org/manuscript/202305.0219/v1) 
and in revision for Bioengineering. To cite: 

O’Reilly, C.; Oruganti, S.D.R.; Tilwani, D.; Bradshaw, J. Model-Driven Analysis of ECG Using Reinforcement Learning. Preprints.org 2023, 2023050219. https://doi.org/10.20944/preprints202305.0219.v1.

<details>
<summary>BibTex entries</summary>

```Bibtex
@article{o2023model,
  title={Model-Driven Analysis of ECG Using Reinforcement Learning},
  author={O’Reilly, Christian and Oruganti, Sai Durga Rithvik and Tilwani, Deepa and Bradshaw, Jessica},
  year={2023},
  publisher={Preprints},
  doi={10.20944/preprints202305.0219.v1},
  pages = {2023050219}
}
```

</details>

## Abstract

Modeling is essential to understand better the generative mechanisms responsible for experimental observations gathered from complex systems. In this work, we are using such an approach to analyze the electrocardiogram (ECG). We present a systematic framework to decompose ECG signals into sums of overlapping lognormal components. We used reinforcement learning to train a deep neural network to estimate the modeling parameters from ECG recorded in babies of 1 to 24 months of age. We demonstrate this model-driven approach by showing how the extracted parameters vary with age. After correction for multiple tests, 10 of 24 modeling parameters showed statistical significance below the 0.01 threshold, with absolute Kendall rank correlation coefficients in the [0.27, 0.51] range. We presented a model-driven approach to the analysis of ECG. The impact of this framework on fundamental science and clinical applications is likely to be increased by further refining the modeling of the physiological mechanisms generating the ECG. By improving the physiological interpretability, this approach can provide a window into latent variables important for understanding the heart-beating process and its control by the autonomous nervous system.
