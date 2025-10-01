# Hierarchical Bayesian Model for Disaggregating Suppressed Census Data

## Overview

This project contains a sophisticated hierarchical Bayesian model designed to estimate the true, unsuppressed counts of household data at a low geographic level (SA2) by leveraging suppressed data from multiple geographic levels (SA2, TA, and National).

The primary challenge is that the input data has been confidentialised. This means some values are suppressed (hidden), and others have been modified through a process of Random Rounding to base 3 (RR3). The model is designed to account for both of these uncertainties.

## Methodology

The final model (`bayesian_model.py`) uses a hierarchical Gibbs sampler to generate posterior distributions for the true counts at each geographic level. This approach was chosen over a simpler deterministic optimization as it better captures the statistical nature of the problem and provides uncertainty estimates for the final counts.

The key features of the model are:

1.  **Hierarchical Structure**: The model respects the geographic hierarchy. The national total informs the distribution of counts across TAs, and each TA's total in turn informs the distribution of counts across its constituent SA2s.

2.  **Gibbs Sampling**: The model uses an iterative Gibbs sampler to explore the space of plausible solutions. Instead of finding a single "best" answer, it generates thousands of samples from the posterior distribution, the average of which is used as the final estimate.

3.  **Custom Sampling for Confidentialised Data**: The model addresses the data's uncertainty not by taking any value as ground truth, but by resampling the "true" integer counts in each iteration using samplers that are precisely tailored to the rules of confidentialisation:

      * **For Suppressed Values**: When a parent TA's value is missing, its value is imputed from a **Poisson distribution** informed by the sum of its children SA2s. Missing child SA2 values are then imputed via a **multinomial distribution** based on their newly sampled parent's total.
      * **For Rounded (RR3) Values**: The model knows the true value must be one of five integers `{R-2, R-1, R, R+1, R+2}` around the rounded value `R`. It imputes a new "truer" value by drawing from a **custom discrete probability distribution**. This distribution is intelligently weighted, assigning a higher probability to the integer that is most likely given the sum of its children SA2 areas.

This robust sampling strategy allows the model to correctly navigate the noise and missingness introduced by the confidentialization process.

### Model Structure and Flow

The model respects the geographic hierarchy: the national total informs the distribution of counts across TAs, and each TA's total in turn informs the distribution of counts across its constituent SA2s.

Let:
* $Y_{i,j}$: The true, unobserved count for SA2 $i$ within TA $j$.
* $X_{j}$: The true, unobserved total count for TA $j$.
* $T$: The true, unobserved National Total.
* $O_{i,j}$: The observed (rounded or suppressed) value for SA2 $i$.
* $O_{j}^{TA}$: The observed (rounded or suppressed) value for TA $j$.

The hierarchy is defined by a series of conditional distributions:

$$
\begin{align*}
P(X_1, \dots, X_J | T, \boldsymbol{\alpha}^{TA}) &\sim \text{Multinomial}(T, \boldsymbol{p}^{TA}) \\
P(Y_{i,j} | X_j, \boldsymbol{\alpha}^{SA2}) &\sim \text{Multinomial}(X_j, \boldsymbol{p}_{j}^{SA2}) \\
\end{align*}
$$

Where $\boldsymbol{p}^{TA}$ and $\boldsymbol{p}_{j}^{SA2}$ are the estimated proportions derived from Dirichlet priors (or $\boldsymbol{\alpha}$ parameters) that are updated in each iteration based on the observed data.

### Custom Sampling for Confidentialised Data

The model addresses the data's uncertainty not by taking any value as ground truth, but by resampling the "true" integer counts in each iteration using samplers that are precisely tailored to the rules of confidentialisation.

#### 1. Imputing Suppressed Values (NaNs)
When a value is suppressed (missing), it is imputed using a Poisson likelihood informed by the current estimate of its sub-components or the overall total.

* **Suppressed TA Total ($O_{j}^{TA} = \text{NaN}$):** The sampled TA total is primarily driven by the **Multinomial draw** from the National total, ensuring the National-level constraint is met.
* **Suppressed SA2 Count ($O_{i,j} = \text{NaN}$):** The SA2 count is determined by the **Multinomial draw** from its newly sampled parent TA's total.

#### 2. Imputing Rounded (RR3) Values
When a value $R$ is observed (rounded), the true value $X$ must be in the set $\{R-2, R-1, R, R+1, R+2\}$. The model imputes a new "truer" value by drawing from a **custom discrete probability distribution**.

* **Constrained TA Total ($O_{j}^{TA} = R$):** The true TA total, $X_j$, is resampled from the possible set of five integers. The probability of choosing a specific integer $k$ is intelligently weighted by a Poisson likelihood, whose mean ($\lambda$) is set to the current sum of its constituent SA2 children ($\sum_{i} Y_{i,j}$).

$$ 
P(X_j = k | \sum_{i} Y_{i,j}, O_{j}^{TA}=R) \propto \text{Poisson}(k | \lambda = \sum_{i} Y_{i,j}) \cdot \mathbb{I}(k \in \{R\pm 2\})
$$

This robust sampling strategy allows the model to correctly navigate the noise and missingness introduced by the confidentialization process.

## File Structure

```
.
├── bayesian_model.py       # Main script with the final hierarchical Bayesian model.
├── run_optimization.py     # Script for an earlier, deterministic optimization-based approach.
├── data/
│   ├── classifications/    # Mapping files (e.g., sa2_to_ta_map.csv).
│   ├── processed/          # Processed input data files.
│   │   ├── sa2_level_data.csv
│   │   ├── ta_level_data.csv
│   │   └── national_level_data.csv
│   └── outputs/            # Directory where all model outputs are saved.
│       ├── bayesian_samples/
│       │   └── ... (raw .npy sample files for each category)
│       └── bayesian_summaries/
│           └── ... (summary .csv files for each category)
└── README.md               # This file.
```

## How to Run the Model

### Dependencies

The script requires the following Python libraries:

  * `pandas`
  * `numpy`
  * `scipy`

You can install them via pip:
`pip install pandas numpy scipy`

### Execution

To run the full analysis, execute the main script from your terminal. The script will automatically discover all unique data categories, process them in a loop, and save the results.

```sh
python bayesian_model.py
```

**Note:** This is a computationally intensive process that will take a significant amount of time to run through all categories.

## Understanding the Output

The script will save two types of files for each category (e.g., `hh_1102_bed_01_...`) into the `data/outputs/` directory:

1.  **Summaries (`bayesian_summaries/` folder)**: A `.csv` file containing the final results. Key columns include:

      * `OBS_VALUE`: The original, suppressed/rounded value.
      * `estimated_count_mean`: The main estimate for the true count (the mean of the posterior samples).
      * `ci_95_lower` & `ci_95_upper`: A 95% credible interval, providing a plausible range for the true count.

2.  **Raw Samples (`bayesian_samples/` folder)**: A `.npy` file containing the raw array of posterior samples from the Gibbs sampler. This can be loaded using `numpy.load()` for more detailed analysis, such as plotting the posterior distribution for a specific SA2.

## Results Visualization

For visual confirmation of sampler convergence and a detailed look at the uncertainty estimates, please see the generated HTML report:

Diagnostic Report Link: [[dylannpaterson.github.io](https://dylannpaterson.github.io/stats/)]

The plot below shows a sample of the model's output for a single category. It compares the original observed values (x-axis) to the model's estimated mean counts (y-axis).

  * **Blue Circles** represent data that was rounded (RR3). The model often adjusts these values to satisfy the geographic constraints.
  * **Green Triangles** (at x=-1) represent data that was suppressed. The model imputes these values and provides an estimate with uncertainty.
  * **Error bars** show the 95% credible interval for each estimate, indicating the model's certainty.
  * The **dashed line** is the `y=x` line, where no change would have been made.
