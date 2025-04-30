# Variant Caller Optimizer

A comprehensive framework for optimizing somatic variant caller parameters using Optuna. This tool supports both tumor-only and tumor-normal paired sample analysis across multiple popular variant callers.

## Features

- **Multi-caller Optimization**: Optimize parameters for MuTect2, VarDict, FreeBayes, and LoFreq
- **Flexible Analysis Modes**: Support for both tumor-only and paired tumor-normal analysis
- **Automatic Parameter Tuning**: Uses Optuna to efficiently search optimal parameter spaces
- **Performance Evaluation**: Evaluates calls against truth sets using industry-standard metrics
- **Visualization**: Generates optimization history and parameter importance plots
- **Extensible Design**: Easy to add additional variant callers or custom evaluation metrics


## Prerequisites

- Python 3.8+
- Variant callers installed and in PATH:
  - GATK (for MuTect2)
  - VarDict
  - FreeBayes
  - LoFreq
- Reference genome
- Input BAM files (tumor +/- normal)
- Optional: Truth VCF for performance evaluation

## Quick Start

```python
from variant_caller_optimizer import VariantCallerOptimizer

# Initialize the optimizer
optimizer = VariantCallerOptimizer(
    reference_genome="/path/to/reference.fa",
    tumor_bam="/path/to/tumor.bam",
    normal_bam="/path/to/normal.bam",  # Optional, set to None for tumor-only mode
    truth_vcf="/path/to/truth.vcf",    # Optional, for evaluation
    output_dir="./optimization_results",
    n_trials=50
)

# Optimize a specific variant caller
best_params = optimizer.optimize_mutect2()
print(f"Best MuTect2 parameters: {best_params}")

# Or optimize all supported variant callers
all_best_params = optimizer.optimize_all()
```

## Detailed Usage

### Configuration Options

```python
optimizer = VariantCallerOptimizer(
    reference_genome="/path/to/reference.fa",
    tumor_bam="/path/to/tumor.bam",
    normal_bam="/path/to/normal.bam",  # Set to None for tumor-only
    truth_vcf="/path/to/truth.vcf",    # Optional
    output_dir="./optimization_results",
    n_trials=50,                       # Number of optimization trials
    n_jobs=-1                          # Parallel jobs (-1 for all cores)
)
```

### Optimizing Individual Callers

```python
# Optimize MuTect2
mutect2_params = optimizer.optimize_mutect2()

# Optimize VarDict
vardict_params = optimizer.optimize_vardict()

# Optimize FreeBayes
freebayes_params = optimizer.optimize_freebayes()

# Optimize LoFreq
lofreq_params = optimizer.optimize_lofreq()
```

### Tumor-Only Mode

```python
# Initialize for tumor-only mode (no normal BAM)
tumor_only_optimizer = VariantCallerOptimizer(
    reference_genome="/path/to/reference.fa",
    tumor_bam="/path/to/tumor.bam",
    normal_bam=None,
    truth_vcf="/path/to/truth.vcf",
    output_dir="./tumor_only_results"
)

# Run optimization
tumor_only_results = tumor_only_optimizer.optimize_all()
```

## Output Structure

```
optimization_results/
├── mutect2/
│   ├── best_params.txt
│   ├── all_trials.csv
│   ├── optimization_history.html
│   ├── param_importances.html
│   └── contour.html
├── vardict/
│   └── ...
├── freebayes/
│   └── ...
├── lofreq/
│   └── ...
└── optimization_summary.txt
```

## Parameter Spaces

Each variant caller has a specific parameter space that is optimized:

### MuTect2
- `tumor_lod`: Minimum log odds threshold for tumor variants
- `min_base_quality`: Minimum base quality required
- `pcr_indel_model`: PCR error model to use
- Additional parameters for paired mode

### VarDict
- `min_allele_freq`: Minimum allele frequency threshold
- `min_mapping_quality`: Minimum mapping quality
- `min_base_quality`: Minimum base quality
- `min_reads`: Minimum supporting reads
- `max_mismatches`: Maximum number of mismatches

### FreeBayes
- `min_alternate_fraction`: Minimum fraction of observations supporting an alternate allele
- `min_base_quality`: Minimum base quality
- `min_mapping_quality`: Minimum mapping quality
- `min_alternate_count`: Minimum number of observations supporting an alternate allele
- `use_best_n_alleles`: Number of alleles to evaluate

### LoFreq
- `min_coverage`: Minimum coverage required
- `min_base_quality`: Minimum base quality
- `significance`: Significance threshold for variants
- `min_alt_count`: Minimum count of alternate alleles
- `max_depth`: Maximum read depth to consider

## Evaluation Metrics

The default optimization target is the F1 score, which balances precision and recall. You can modify the evaluation function to prioritize different metrics based on your requirements.

## Extending the Framework

### Adding a New Variant Caller

To add support for a new variant caller:

1. Create a new `optimize_<caller_name>` method in the `VariantCallerOptimizer` class
2. Define the parameter space in the objective function
3. Implement the command execution logic
4. Add the new caller to the `optimize_all` method

### Custom Evaluation Metrics

Modify the `_evaluate_vcf` method to use different evaluation tools or metrics.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [Optuna](https://optuna.org/) for the hyperparameter optimization framework
- The developers of MuTect2, VarDict, FreeBayes, and LoFreq


