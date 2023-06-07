# Model-driven analysis of ECG using reinforcement learning

This repository contains the code used for the paper entitled "Model-driven analysis of ECG using reinforcement learning" published at [Bioengineering](https://www.mdpi.com/2306-5354/10/6/696). To cite: 

O’Reilly C, Oruganti SDR, Tilwani D, Bradshaw J. Model-Driven Analysis of ECG Using Reinforcement Learning. Bioengineering. 2023; 10(6):696. https://doi.org/10.3390/bioengineering10060696

<details>
<summary>BibTex entries</summary>

```Bibtex
@Article{bioengineering10060696,
AUTHOR = {O’Reilly, Christian and Oruganti, Sai Durga Rithvik and Tilwani, Deepa and Bradshaw, Jessica},
TITLE = {Model-Driven Analysis of ECG Using Reinforcement Learning},
JOURNAL = {Bioengineering},
VOLUME = {10},
YEAR = {2023},
NUMBER = {6},
ARTICLE-NUMBER = {696},
URL = {https://www.mdpi.com/2306-5354/10/6/696},
ISSN = {2306-5354},
ABSTRACT = {Modeling is essential to better understand the generative mechanisms responsible for experimental observations gathered from complex systems. In this work, we are using such an approach to analyze the electrocardiogram (ECG). We present a systematic framework to decompose ECG signals into sums of overlapping lognormal components. We use reinforcement learning to train a deep neural network to estimate the modeling parameters from an ECG recorded in babies from 1 to 24 months of age. We demonstrate this model-driven approach by showing how the extracted parameters vary with age. From the 751,510 PQRST complexes modeled, 82.7% provided a signal-to-noise ratio that was sufficient for further analysis (&gt;5 dB). After correction for multiple tests, 10 of the 24 modeling parameters exhibited statistical significance below the 0.01 threshold, with absolute Kendall rank correlation coefficients in the [0.27, 0.51] range. These results confirm that this model-driven approach can capture sensitive ECG parameters. Due to its physiological interpretability, this approach can provide a window into latent variables which are important for understanding the heart-beating process and its control by the autonomous nervous system.},
DOI = {10.3390/bioengineering10060696}
}
```

</details>

## Abstract

Modeling is essential to better understand the generative mechanisms responsible for experimental observations gathered from complex systems. In this work, we are using such an approach to analyze the electrocardiogram (ECG). We present a systematic framework to decompose ECG signals into sums of overlapping lognormal components. We use reinforcement learning to train a deep neural network to estimate the modeling parameters from an ECG recorded in babies from 1 to 24 months of age. We demonstrate this model-driven approach by showing how the extracted parameters vary with age. From the 751,510 PQRST complexes modeled, 82.7% provided a signal-to-noise ratio that was sufficient for further analysis (&gt;5 dB). After correction for multiple tests, 10 of the 24 modeling parameters exhibited statistical significance below the 0.01 threshold, with absolute Kendall rank correlation coefficients in the [0.27, 0.51] range. These results confirm that this model-driven approach can capture sensitive ECG parameters. Due to its physiological interpretability, this approach can provide a window into latent variables which are important for understanding the heart-beating process and its control by the autonomous nervous system.
