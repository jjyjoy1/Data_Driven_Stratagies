#!/usr/bin/env python3
# GATK Germline Variant Pipeline Optimization with Optuna (Stage III)
# Focus on Variant Filtration parameters optimization using NA12878 subsample

import os
import sys
import subprocess
import optuna
import pandas as pd
import numpy as np
import time
import logging
import shutil
import joblib
from pathlib import Path
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gatk_stage3_optimization.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("GATK-Optuna-Stage3")

# Load configuration
def load_config(config_path="stage3_config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load Stage III specific configuration
def load_stage3_config(config_path="stage3_config.yaml"):
    with open(config_path, 'r') as f:
        stage3_config = yaml.safe_load(f)
    return stage3_config

config = load_config()
stage3_config = load_stage3_config()

# Configuration class with paths and settings
class Config:
    # Paths
    REFERENCE_GENOME = config["reference"]["genome"]
    OUTPUT_DIR = os.path.join(config["output"]["directory"], "stage3")
    
    # Input files prepared from Stage II
    RAW_VCF = os.path.join(config["output"]["directory"], "stage3_prep", "raw_variants.vcf")
    RAW_SNPS_VCF = os.path.join(config["output"]["directory"], "stage3_prep", "raw_snps.vcf")
    RAW_INDELS_VCF = os.path.join(config["output"]["directory"], "stage3_prep", "raw_indels.vcf")
    
    # Truth set for evaluation
    GIAB_VCF = config["truth_sets"]["GIAB"]["vcf"]
    GIAB_BED = config["truth_sets"]["GIAB"]["bed"]
    RTG_SDF_DIR = config["truth_sets"]["GIAB"]["rtg_sdf"]
    
    # Tools
    GATK = "gatk"
    BGZIP = "bgzip"
    TABIX = "tabix"
    RTGTOOLS = "rtg"
    
    # Number of CPU cores to use
    THREADS = config["computing"]["threads"]
    
    # Optimization settings
    N_TRIALS = config["optuna"]["n_trials"]
    TIMEOUT = config["optuna"]["timeout"]

# Helper Functions
def run_command(cmd, desc=None, check=True):
    """Run a shell command and log it."""
    if desc:
        logger.info(f"Running: {desc}")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, check=check, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    logger.debug(f"Took {elapsed:.2f} seconds")
    if result.stderr:
        logger.debug(f"STDERR: {result.stderr}")
    
    return result.stdout, result.stderr, result.returncode

def apply_snp_filters(input_vcf, output_vcf, filter_params):
    """Apply SNP filtering with trial parameters."""
    # Extract filter parameters
    qd_filter = filter_params["QD"]
    fs_filter = filter_params["FS"]
    mq_filter = filter_params["MQ"]
    sor_filter = filter_params.get("SOR", 3.0)  # Default if not specified
    mqrs_filter = filter_params.get("MQRankSum", -12.5)  # Default if not specified
    rprs_filter = filter_params.get("ReadPosRankSum", -8.0)  # Default if not specified
    
    # Build the filter expression for SNPs
    filter_expr = (
        f"QD < {qd_filter} || "
        f"FS > {fs_filter} || "
        f"MQ < {mq_filter}"
    )
    
    # Add optional filters if specified
    if "SOR" in filter_params:
        filter_expr += f" || SOR > {sor_filter}"
    
    if "MQRankSum" in filter_params:
        filter_expr += f" || MQRankSum < {mqrs_filter}"
    
    if "ReadPosRankSum" in filter_params:
        filter_expr += f" || ReadPosRankSum < {rprs_filter}"
    
    # Build GATK command
    cmd = [
        Config.GATK, "VariantFiltration",
        "-R", Config.REFERENCE_GENOME,
        "-V", input_vcf,
        "-O", output_vcf,
        "--filter-expression", filter_expr,
        "--filter-name", "SNP_FILTER"
    ]
    
    logger.info(f"Applying SNP filters: QD={qd_filter}, FS={fs_filter}, MQ={mq_filter}, "
               f"SOR={sor_filter}, MQRankSum={mqrs_filter}, ReadPosRankSum={rprs_filter}")
    
    run_command(cmd, "Applying SNP filters")
    
    return output_vcf

def apply_indel_filters(input_vcf, output_vcf, filter_params):
    """Apply INDEL filtering with trial parameters."""
    # Extract filter parameters
    qd_filter = filter_params["QD"]
    fs_filter = filter_params["FS"]
    sor_filter = filter_params.get("SOR", 10.0)  # Default if not specified
    rprs_filter = filter_params.get("ReadPosRankSum", -20.0)  # Default if not specified
    
    # Build the filter expression for INDELs
    filter_expr = f"QD < {qd_filter} || FS > {fs_filter}"
    
    # Add optional filters if specified
    if "SOR" in filter_params:
        filter_expr += f" || SOR > {sor_filter}"
    
    if "ReadPosRankSum" in filter_params:
        filter_expr += f" || ReadPosRankSum < {rprs_filter}"
    
    # Build GATK command
    cmd = [
        Config.GATK, "VariantFiltration",
        "-R", Config.REFERENCE_GENOME,
        "-V", input_vcf,
        "-O", output_vcf,
        "--filter-expression", filter_expr,
        "--filter-name", "INDEL_FILTER"
    ]
    
    logger.info(f"Applying INDEL filters: QD={qd_filter}, FS={fs_filter}, "
               f"SOR={sor_filter}, ReadPosRankSum={rprs_filter}")
    
    run_command(cmd, "Applying INDEL filters")
    
    return output_vcf

def select_passing_variants(input_vcf, output_vcf):
    """Select only PASS variants after filtering."""
    cmd = [
        Config.GATK, "SelectVariants",
        "-R", Config.REFERENCE_GENOME,
        "-V", input_vcf,
        "-O", output_vcf,
        "--exclude-filtered"
    ]
    
    run_command(cmd, "Selecting PASS variants")
    
    return output_vcf

def merge_filtered_vcfs(snp_vcf, indel_vcf, output_vcf):
    """Merge filtered SNP and INDEL VCFs."""
    cmd = [
        Config.GATK, "MergeVcfs",
        "-I", snp_vcf,
        "-I", indel_vcf,
        "-O", output_vcf
    ]
    
    run_command(cmd, "Merging filtered VCFs")
    
    return output_vcf

def evaluate_variants(vcf_file, trial_number):
    """Evaluate filtered variants against GIAB truth set using RTG Tools."""
    eval_dir = os.path.join(Config.OUTPUT_DIR, "evaluation", f"trial_{trial_number}")
    os.makedirs(eval_dir, exist_ok=True)
    
    cmd = [
        Config.RTGTOOLS, "vcfeval",
        "-b", Config.GIAB_VCF,
        "-c", vcf_file,
        "-e", Config.GIAB_BED,
        "-t", Config.RTG_SDF_DIR,
        "-o", eval_dir,
        "--ref-overlap",
        "--all-records",
        "--output-mode", "split"  # Split results by variant type (SNP/INDEL)
    ]
    
    run_command(cmd, "Evaluating variants with RTG vcfeval")
    
    # Parse metrics from summary files
    metrics = {}
    
    # Overall metrics
    summary_file = os.path.join(eval_dir, "summary.txt")
    with open(summary_file, 'r') as f:
        for line in f:
            if line.startswith("F-measure"):
                metrics["f_measure"] = float(line.split()[-1])
            elif line.startswith("Precision"):
                metrics["precision"] = float(line.split()[-1])
            elif line.startswith("Recall"):
                metrics["recall"] = float(line.split()[-1])
    
    # SNP-specific metrics
    snp_summary = os.path.join(eval_dir, "snp_summary.txt")
    if os.path.exists(snp_summary):
        with open(snp_summary, 'r') as f:
            for line in f:
                if line.startswith("F-measure"):
                    metrics["snp_f_measure"] = float(line.split()[-1])
                elif line.startswith("Precision"):
                    metrics["snp_precision"] = float(line.split()[-1])
                elif line.startswith("Recall"):
                    metrics["snp_recall"] = float(line.split()[-1])
    
    # INDEL-specific metrics
    indel_summary = os.path.join(eval_dir, "indel_summary.txt")
    if os.path.exists(indel_summary):
        with open(indel_summary, 'r') as f:
            for line in f:
                if line.startswith("F-measure"):
                    metrics["indel_f_measure"] = float(line.split()[-1])
                elif line.startswith("Precision"):
                    metrics["indel_precision"] = float(line.split()[-1])
                elif line.startswith("Recall"):
                    metrics["indel_recall"] = float(line.split()[-1])
    
    # Calculate combined weighted metric if applicable
    if "snp_f_measure" in metrics and "indel_f_measure" in metrics:
        w_snp = config["evaluation"]["weights"]["snp_f_measure"]
        w_indel = config["evaluation"]["weights"]["indel_f_measure"]
        
        combined_f = ((w_snp * metrics["snp_f_measure"]) + 
                       (w_indel * metrics["indel_f_measure"])) / (w_snp + w_indel)
        
        metrics["combined_f_measure"] = combined_f
        logger.info(f"Combined weighted F-measure: {combined_f:.4f}")
    
    return metrics

def objective(trial):
    """Optuna objective function for variant filtration parameter optimization."""
    # Generate trial parameters for SNP filters
    snp_params = {
        "QD": trial.suggest_float("snp_QD", 
                                 stage3_config["parameter_ranges"]["snp_filters"]["QD"]["min"],
                                 stage3_config["parameter_ranges"]["snp_filters"]["QD"]["max"]),
        
        "FS": trial.suggest_float("snp_FS", 
                                 stage3_config["parameter_ranges"]["snp_filters"]["FS"]["min"],
                                 stage3_config["parameter_ranges"]["snp_filters"]["FS"]["max"]),
        
        "MQ": trial.suggest_float("snp_MQ", 
                                 stage3_config["parameter_ranges"]["snp_filters"]["MQ"]["min"],
                                 stage3_config["parameter_ranges"]["snp_filters"]["MQ"]["max"])
    }
    
    # Add optional SNP filters if specified in config
    if "SOR" in stage3_config["parameter_ranges"]["snp_filters"]:
        snp_params["SOR"] = trial.suggest_float("snp_SOR", 
                                              stage3_config["parameter_ranges"]["snp_filters"]["SOR"]["min"],
                                              stage3_config["parameter_ranges"]["snp_filters"]["SOR"]["max"])
    
    if "MQRankSum" in stage3_config["parameter_ranges"]["snp_filters"]:
        snp_params["MQRankSum"] = trial.suggest_float("snp_MQRankSum", 
                                                    stage3_config["parameter_ranges"]["snp_filters"]["MQRankSum"]["min"],
                                                    stage3_config["parameter_ranges"]["snp_filters"]["MQRankSum"]["max"])
    
    if "ReadPosRankSum" in stage3_config["parameter_ranges"]["snp_filters"]:
        snp_params["ReadPosRankSum"] = trial.suggest_float("snp_ReadPosRankSum", 
                                                         stage3_config["parameter_ranges"]["snp_filters"]["ReadPosRankSum"]["min"],
                                                         stage3_config["parameter_ranges"]["snp_filters"]["ReadPosRankSum"]["max"])
    
    # Generate trial parameters for INDEL filters
    indel_params = {
        "QD": trial.suggest_float("indel_QD", 
                                 stage3_config["parameter_ranges"]["indel_filters"]["QD"]["min"],
                                 stage3_config["parameter_ranges"]["indel_filters"]["QD"]["max"]),
        
        "FS": trial.suggest_float("indel_FS", 
                                 stage3_config["parameter_ranges"]["indel_filters"]["FS"]["min"],
                                 stage3_config["parameter_ranges"]["indel_filters"]["FS"]["max"])
    }
    
    # Add optional INDEL filters if specified in config
    if "SOR" in stage3_config["parameter_ranges"]["indel_filters"]:
        indel_params["SOR"] = trial.suggest_float("indel_SOR", 
                                                stage3_config["parameter_ranges"]["indel_filters"]["SOR"]["min"],
                                                stage3_config["parameter_ranges"]["indel_filters"]["SOR"]["max"])
    
    if "ReadPosRankSum" in stage3_config["parameter_ranges"]["indel_filters"]:
        indel_params["ReadPosRankSum"] = trial.suggest_float("indel_ReadPosRankSum", 
                                                         stage3_config["parameter_ranges"]["indel_filters"]["ReadPosRankSum"]["min"],
                                                         stage3_config["parameter_ranges"]["indel_filters"]["ReadPosRankSum"]["max"])
    
    logger.info(f"Trial {trial.number} - SNP filters: {snp_params}")
    logger.info(f"Trial {trial.number} - INDEL filters: {indel_params}")
    
    # Output files for this trial
    trial_dir = os.path.join(Config.OUTPUT_DIR, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)
    
    try:
        # Time the filtering pipeline
        start_time = time.time()
        
        # Apply SNP filters
        filtered_snps_vcf = os.path.join(trial_dir, "filtered_snps.vcf")
        apply_snp_filters(Config.RAW_SNPS_VCF, filtered_snps_vcf, snp_params)
        
        # Apply INDEL filters
        filtered_indels_vcf = os.path.join(trial_dir, "filtered_indels.vcf")
        apply_indel_filters(Config.RAW_INDELS_VCF, filtered_indels_vcf, indel_params)
        
        # Select passing variants
        passing_snps_vcf = os.path.join(trial_dir, "passing_snps.vcf")
        select_passing_variants(filtered_snps_vcf, passing_snps_vcf)
        
        passing_indels_vcf = os.path.join(trial_dir, "passing_indels.vcf")
        select_passing_variants(filtered_indels_vcf, passing_indels_vcf)
        
        # Merge filtered VCFs
        merged_vcf = os.path.join(trial_dir, "merged_filtered.vcf")
        merge_filtered_vcfs(passing_snps_vcf, passing_indels_vcf, merged_vcf)
        
        # Compress and index the merged VCF if needed
        if not merged_vcf.endswith(".gz"):
            bgzip_cmd = [Config.BGZIP, merged_vcf]
            run_command(bgzip_cmd, "Compressing VCF")
            merged_vcf += ".gz"
        
        tabix_cmd = [Config.TABIX, "-p", "vcf", merged_vcf]
        run_command(tabix_cmd, "Indexing VCF")
        
        # Evaluate the filtered variants
        metrics = evaluate_variants(merged_vcf, trial.number)
        
        # Calculate runtime
        runtime = time.time() - start_time
        
        # Determine which metric to use for optimization
        if config["evaluation"]["primary_metric"] == "combined" and "combined_f_measure" in metrics:
            primary_metric_value = metrics["combined_f_measure"]
        else:
            primary_metric_value = metrics.get(config["evaluation"]["primary_metric"], 0)
        
        # Apply runtime penalty if configured
        if config["evaluation"]["weights"]["runtime"] > 0:
            runtime_weight = config["evaluation"]["weights"]["runtime"]
            max_runtime = config["evaluation"]["max_runtime"]
            runtime_penalty = min(1.0, runtime / max_runtime) * runtime_weight
            primary_metric_value = primary_metric_value * (1 - runtime_penalty)
        
        # Log results
        logger.info(f"Trial {trial.number} completed - Primary metric: {primary_metric_value:.4f}, "
                  f"F-measure: {metrics.get('f_measure', 0):.4f}, "
                  f"SNP F-measure: {metrics.get('snp_f_measure', 0):.4f}, "
                  f"INDEL F-measure: {metrics.get('indel_f_measure', 0):.4f}, "
                  f"Runtime: {runtime:.2f}s")
        
        # Save trial details to CSV
        trial_results = {
            'trial_number': trial.number,
            'runtime': runtime
        }
        
        # Add all filter parameters
        for param, value in snp_params.items():
            trial_results[f'snp_{param}'] = value
        
        for param, value in indel_params.items():
            trial_results[f'indel_{param}'] = value
        
        # Add all metrics
        for metric, value in metrics.items():
            trial_results[metric] = value
        
        df = pd.DataFrame([trial_results])
        results_csv = os.path.join(Config.OUTPUT_DIR, "trial_results.csv")
        
        # Append to CSV or create a new one
        if os.path.exists(results_csv):
            df.to_csv(results_csv, mode='a', header=False, index=False)
        else:
            df.to_csv(results_csv, index=False)
        
        return primary_metric_value
    
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {str(e)}")
        return 0.0  # Return poorest score on error

def main():
    """Main function to run the optimization."""
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    logger.info("Starting GATK germline variant pipeline optimization - Stage III (Variant Filtration)")
    
    # Check if raw VCF files exist
    for vcf_file in [Config.RAW_VCF, Config.RAW_SNPS_VCF, Config.RAW_INDELS_VCF]:
        if not os.path.exists(vcf_file):
            logger.error(f"Required input file not found: {vcf_file}")
            logger.error("Please run the prepare_stage3.py script first")
            sys.exit(1)
    
    # Create Optuna study
    logger.info(f"Creating Optuna study with {Config.N_TRIALS} trials")
    
    # Set up pruner if enabled
    pruner = None
    if config["optuna"]["pruner"] == "MedianPruner":
        pruner = optuna.pruners.MedianPruner()
    elif config["optuna"]["pruner"] == "PercentilePruner":
        pruner = optuna.pruners.PercentilePruner(25.0)
    
    # Set up sampler
    sampler = None
    if config["optuna"]["sampler"] == "TPESampler":
        sampler = optuna.samplers.TPESampler(seed=config["optuna"]["seed"])
    elif config["optuna"]["sampler"] == "RandomSampler":
        sampler = optuna.samplers.RandomSampler(seed=config["optuna"]["seed"])
    elif config["optuna"]["sampler"] == "CmaEsSampler":
        sampler = optuna.samplers.CmaEsSampler(seed=config["optuna"]["seed"])
    
    study = optuna.create_study(
        study_name="gatk_variant_filtration_optimization",
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )
    
    # Run optimization
    study.optimize(objective, n_trials=Config.N_TRIALS, timeout=Config.TIMEOUT)
    
    # Report best parameters
    logger.info(f"Optimization completed. Best value: {study.best_value:.4f}")
    logger.info(f"Best parameters:")
    
    # Separate SNP and INDEL filter parameters
    best_snp_filters = {}
    best_indel_filters = {}
    
    for param_name, param_value in study.best_params.items():
        if param_name.startswith("snp_"):
            actual_param = param_name.replace("snp_", "")
            best_snp_filters[actual_param] = param_value
        elif param_name.startswith("indel_"):
            actual_param = param_name.replace("indel_", "")
            best_indel_filters[actual_param] = param_value
    
    logger.info(f"Best SNP filters: {best_snp_filters}")
    logger.info(f"Best INDEL filters: {best_indel_filters}")
    
    # Save best parameters as YAML
    best_params = {
        "variant_filtration": {
            "snp_filters": best_snp_filters,
            "indel_filters": best_indel_filters
        }
    }
    
    with open(os.path.join(Config.OUTPUT_DIR, "stage3_best_params.yaml"), 'w') as f:
        yaml.dump(best_params, f, default_flow_style=False)
    
    # Save study
    joblib.dump(study, os.path.join(Config.OUTPUT_DIR, "study.pkl"))
    
    # Generate optimization visualization
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour
        import matplotlib.pyplot as plt
        
        # Plot optimization history
        fig = plot_optimization_history(study)
        fig.write_image(os.path.join(Config.OUTPUT_DIR, "optimization_history.png"))
        
        # Plot parameter importances
        fig = plot_param_importances(study)
        fig.write_image(os.path.join(Config.OUTPUT_DIR, "param_importances.png"))
        
        # Plot contour plots for key parameters
        # SNP parameters
        snp_params = [p for p in study.best_params.keys() if p.startswith("snp_")]
        if len(snp_params) >= 2:
            for i in range(len(snp_params)):
                for j in range(i+1, len(snp_params)):
                    param1, param2 = snp_params[i], snp_params[j]
                    fig = plot_contour(study, params=[param1, param2])
                    fig.write_image(os.path.join(Config.OUTPUT_DIR, f"contour_{param1}_vs_{param2}.png"))
        
        # INDEL parameters
        indel_params = [p for p in study.best_params.keys() if p.startswith("indel_")]
        if len(indel_params) >= 2:
            for i in range(len(indel_params)):
                for j in range(i+1, len(indel_params)):
                    param1, param2 = indel_params[i], indel_params[j]
                    fig = plot_contour(study, params=[param1, param2])
                    fig.write_image(os.path.join(Config.OUTPUT_DIR, f"contour_{param1}_vs_{param2}.png"))
    except Exception as e:
        logger.warning(f"Could not generate visualizations: {str(e)}")
    
    # Generate final optimized variant filtering pipeline
    logger.info("Creating final optimized variant filtering pipeline script")
    
    script_content = f"""#!/bin/bash
# Optimized GATK Variant Filtration Pipeline
# Generated from Optuna optimization

set -e

# Input raw VCF
RAW_VCF="$1"
OUTPUT_PREFIX="$2"
REFERENCE_GENOME="{Config.REFERENCE_GENOME}"

if [ -z "$RAW_VCF" ] || [ -z "$OUTPUT_PREFIX" ]; then
    echo "Usage: $0 <raw_vcf> <output_prefix>"
    exit 1
fi

# Extract SNPs and INDELs
{Config.GATK} SelectVariants \\
    -R $REFERENCE_GENOME \\
    -V $RAW_VCF \\
    -O ${{OUTPUT_PREFIX}}_snps.vcf \\
    --select-type-to-include SNP

{Config.GATK} SelectVariants \\
    -R $REFERENCE_GENOME \\
    -V $RAW_VCF \\
    -O ${{OUTPUT_PREFIX}}_indels.vcf \\
    --select-type-to-include INDEL

# Apply optimized SNP filters
{Config.GATK} VariantFiltration \\
    -R $REFERENCE_GENOME \\
    -V ${{OUTPUT_PREFIX}}_snps.vcf \\
    -O ${{OUTPUT_PREFIX}}_filtered_snps.vcf \\
    --filter-expression "{' || '.join([f'{k} < {v}' if k != 'FS' else f'{k} > {v}' for k, v in best_snp_filters.items()])}" \\
    --filter-name "SNP_FILTER"

# Apply optimized INDEL filters
{Config.GATK} VariantFiltration \\
    -R $REFERENCE_GENOME \\
    -V ${{OUTPUT_PREFIX}}_indels.vcf \\
    -O ${{OUTPUT_PREFIX}}_filtered_indels.vcf \\
    --filter-expression "{' || '.join([f'{k} < {v}' if k != 'FS' else f'{k} > {v}' for k, v in best_indel_filters.items()])}" \\
    --filter-name "INDEL_FILTER"

# Select passing variants
{Config.GATK} SelectVariants \\
    -R $REFERENCE_GENOME \\
    -V ${{OUTPUT_PREFIX}}_filtered_snps.vcf \\
    -O ${{OUTPUT_PREFIX}}_passing_snps.vcf \\
    --exclude-filtered

{Config.GATK} SelectVariants \\
    -R $REFERENCE_GENOME \\
    -V ${{OUTPUT_PREFIX}}_filtered_indels.vcf \\
    -O ${{OUTPUT_PREFIX}}_passing_indels.vcf \\
    --exclude-filtered

# Merge filtered VCFs
{Config.GATK} MergeVcfs \\
    -I ${{OUTPUT_PREFIX}}_passing_snps.vcf \\
    -I ${{OUTPUT_PREFIX}}_passing_indels.vcf \\
    -O ${{OUTPUT_PREFIX}}_filtered.vcf

# Compress and index
{Config.BGZIP} ${{OUTPUT_PREFIX}}_filtered.vcf
{Config.TABIX} -p vcf ${{OUTPUT_PREFIX}}_filtered.vcf.gz

echo "Pipeline completed successfully!"
echo "Final filtered VCF: ${{OUTPUT_PREFIX}}_filtered.vcf.gz"
"""
    
    with open(os.path.join(Config.OUTPUT_DIR, "optimized_filter_pipeline.sh"), 'w') as f:
        f.write(script_content)
    
    os.chmod(os.path.join(Config.OUTPUT_DIR, "optimized_filter_pipeline.sh"), 0o755)
    
    logger.info(f"Optimized pipeline script created: {os.path.join(Config.OUTPUT_DIR, 'optimized_filter_pipeline.sh')}")
    logger.info("Stage III optimization pipeline completed successfully")

if __name__ == "__main__":
    main()


