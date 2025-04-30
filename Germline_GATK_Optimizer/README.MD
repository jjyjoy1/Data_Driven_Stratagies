# GATK-Optuna-Pipeline-Optimization

A multi-stage optimization framework for GATK-based germline variant calling pipelines using Optuna.

## Overview

This repository contains a comprehensive framework for optimizing the parameters of GATK-based germline variant calling pipelines using Bayesian optimization via Optuna. The framework breaks down the complex parameter space into three sequential optimization stages, making the problem more tractable and efficient.

![Optimization Process Overview](https://via.placeholder.com/800x400.png?text=GATK+Pipeline+Optimization+Framework)

## Key Features

- **Multi-stage optimization** approach to tackle the high-dimensional parameter space
- **Bayesian optimization** using Optuna for efficient parameter tuning
- **Automated benchmarking** against gold standard truth sets (GIAB)
- **Detailed metrics** for both SNPs and INDELs
- **Visualization tools** for optimization history and parameter importance
- **Resource-efficient design** with pruning of poor-performing trials
- **Modular architecture** that can be adapted to different pipelines

## Optimization Stages

### Stage I: Alignment Optimization

Focuses on BWA-MEM and duplicate marking parameters:
- Seed length (`-k`)
- Match/mismatch scores (`-A`, `-B`)
- Gap penalties (`-O`, `-E`)
- Optical duplicate pixel distance

### Stage II: Variant Calling Optimization

Optimizes GATK HaplotypeCaller parameters using the best alignment parameters from Stage I:
- Heterozygosity (SNP and INDEL)
- PCR indel model selection
- Pruning thresholds
- Minimum confidence thresholds

### Stage III: Variant Filtration Optimization

Tunes hard filtering thresholds for SNPs and INDELs separately:
- Quality by Depth (QD)
- FisherStrand (FS)
- MappingQuality (MQ)
- StrandOddsRatio (SOR)
- Read position and mapping quality rank sum tests

## Requirements

- Python 3.6+
- GATK 4.2+
- BWA-MEM
- Samtools
- Picard
- RTG Tools (for vcfeval)
- seqtk (for subsampling)

### Python Dependencies

```
optuna>=2.10.0
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
plotly>=5.3.0
kaleido>=0.2.1
joblib>=1.0.0
pyyaml>=5.4.0
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GATK-Optuna-Pipeline-Optimization.git
cd GATK-Optuna-Pipeline-Optimization

# Create virtual environment
python -m venv gatk_optuna_env
source gatk_optuna_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run setup script
bash setup.sh
```

## Usage

### 1. Configure Your Environment

Edit the `config.yaml` file with your paths and settings:

```yaml
# Paths to reference genome, input files, etc.
reference:
  genome: "/path/to/reference/genome.fa"

samples:
  NA12878:
    fastq_r1: "/path/to/NA12878_R1.fastq.gz"
    fastq_r2: "/path/to/NA12878_R2.fastq.gz"
    
# Optimization settings
optuna:
  n_trials: 50
  timeout: 86400  # 24 hours
```

### 2. Run the Optimization Stages

```bash
# Stage I: Alignment optimization
python gatk-optuna-stage1.py

# Stage II: HaplotypeCaller optimization
python gatk-optuna-stage2.py

# Prepare for Stage III
python prepare-stage3.py

# Stage III: Variant filtration optimization
python gatk-optuna-stage3.py
```

### 3. Apply Optimized Parameters

After completing all stages, an optimized pipeline script will be generated:

```bash
# Run the optimized pipeline on new samples
bash output/stage3/optimized_filter_pipeline.sh raw_variants.vcf output_prefix
```

## Results Interpretation

Each stage produces:
- YAML file with best parameters
- CSV with metrics for all trials
- Optimization plots (history, parameter importance, contour plots)
- Pickled Optuna study object for further analysis

Example visualization of parameter importance:
![Parameter Importance Example](https://via.placeholder.com/600x400.png?text=Parameter+Importance+Plot)

## Adapting for Other Pipelines

This framework can be adapted for:
- Somatic variant calling (Mutect2, Strelka2, etc.)
- Structural variant detection
- RNA-seq analysis pipelines
- And more!

See the `examples/` directory for adaptation examples.

## Advanced Configuration

For more advanced configuration options, see the [Wiki](https://github.com/yourusername/GATK-Optuna-Pipeline-Optimization/wiki).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this framework in your research, please cite:

```
@software{gatk_optuna_pipeline_optimization,
  author = {Your Name},
  title = {GATK-Optuna-Pipeline-Optimization: Multi-stage optimization framework for GATK-based germline variant calling pipelines},
  year = {2025},
  url = {https://github.com/yourusername/GATK-Optuna-Pipeline-Optimization}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Broad Institute for GATK
- Preferred Networks for Optuna
- Genome in a Bottle Consortium for truth sets
