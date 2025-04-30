#!/usr/bin/env python3
# GATK Germline Variant Pipeline Optimization with Optuna (Stage II)
# Focus on HaplotypeCaller parameters optimization using NA12878 subsample

import os
import sys
import subprocess
import tempfile
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
        logging.FileHandler("gatk_stage2_optimization.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("GATK-Optuna-Stage2")

# Load configuration
def load_config(config_path="stage2_config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config()

# Load best parameters from Stage I
def load_stage1_results(stage1_results_path="stage1_best_params.yaml"):
    with open(stage1_results_path, 'r') as f:
        stage1_params = yaml.safe_load(f)
    return stage1_params

stage1_params = load_stage1_results()

# Configuration class with paths and settings
class Config:
    # Paths
    REFERENCE_GENOME = config["reference"]["genome"]
    NA12878_FASTQ_R1 = config["samples"]["NA12878"]["fastq_r1"]
    NA12878_FASTQ_R2 = config["samples"]["NA12878"]["fastq_r2"]
    GIAB_VCF = config["truth_sets"]["GIAB"]["vcf"]
    GIAB_BED = config["truth_sets"]["GIAB"]["bed"]
    OUTPUT_DIR = os.path.join(config["output"]["directory"], "stage2")
    STAGE1_ALIGNED_BAM = os.path.join(config["output"]["directory"], "stage1", "best_aligned.bam")
    
    # Tools
    BWA = "bwa"
    SAMTOOLS = "samtools"
    GATK = "gatk"
    PICARD = "picard"
    BGZIP = "bgzip"
    TABIX = "tabix"
    RTGTOOLS = "rtg"
    
    # Number of CPU cores to use
    THREADS = config["computing"]["threads"]
    
    # Subsampling percentage (0-1)
    SUBSAMPLE_RATIO = config["subsampling"]["ratio"]
    
    # RTG Tools evaluation settings
    RTG_SDF_DIR = config["truth_sets"]["GIAB"]["rtg_sdf"]
    
    # Optimization settings
    N_TRIALS = config["optuna"]["n_trials"]
    TIMEOUT = config["optuna"]["timeout"]
    
    # Best BWA-MEM parameters from Stage I
    BWA_SEED_LENGTH = stage1_params["bwa_mem"]["seed_length"]
    BWA_MATCH_SCORE = stage1_params["bwa_mem"]["match_score"]
    BWA_MISMATCH_PENALTY = stage1_params["bwa_mem"]["mismatch_penalty"]
    BWA_GAP_OPEN_PENALTY = stage1_params["bwa_mem"]["gap_open_penalty"]
    BWA_GAP_EXTENSION_PENALTY = stage1_params["bwa_mem"]["gap_extension_penalty"]
    OPTICAL_DUPLICATE_PIXEL_DISTANCE = stage1_params["mark_duplicates"]["optical_duplicate_pixel_distance"]

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

def prepare_input_bam():
    """
    Check if we already have a best aligned BAM from Stage I,
    if not, create one using the best parameters.
    """
    if os.path.exists(Config.STAGE1_ALIGNED_BAM):
        logger.info(f"Using best aligned BAM from Stage I: {Config.STAGE1_ALIGNED_BAM}")
        return Config.STAGE1_ALIGNED_BAM
    
    logger.info("Best aligned BAM from Stage I not found, creating one with best parameters...")
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Prepare subsampled data first
    r1_out, r2_out = prepare_subsampled_data()
    
    # Run alignment with best parameters from Stage I
    aligned_bam = os.path.join(Config.OUTPUT_DIR, "best_aligned.bam")
    
    run_alignment(r1_out, r2_out, aligned_bam, {
        'trial_number': 'best',
        'seed_length': Config.BWA_SEED_LENGTH,
        'match_score': Config.BWA_MATCH_SCORE,
        'mismatch_penalty': Config.BWA_MISMATCH_PENALTY,
        'gap_open_penalty': Config.BWA_GAP_OPEN_PENALTY,
        'gap_extension_penalty': Config.BWA_GAP_EXTENSION_PENALTY,
        'optical_duplicate_pixel_distance': Config.OPTICAL_DUPLICATE_PIXEL_DISTANCE
    })
    
    return aligned_bam

def prepare_subsampled_data():
    """Prepare subsampled FASTQ files for NA12878."""
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    sample_dir = os.path.join(Config.OUTPUT_DIR, "subsampled")
    os.makedirs(sample_dir, exist_ok=True)
    
    r1_out_gz = os.path.join(sample_dir, "NA12878_sub_R1.fastq.gz")
    r2_out_gz = os.path.join(sample_dir, "NA12878_sub_R2.fastq.gz")
    
    # Check if subsampled files already exist
    if os.path.exists(r1_out_gz) and os.path.exists(r2_out_gz):
        logger.info("Using existing subsampled FASTQ files")
        return r1_out_gz, r2_out_gz
    
    # Subsample FASTQ files
    logger.info(f"Subsampling FASTQ files to {Config.SUBSAMPLE_RATIO*100}%")
    r1_out = os.path.join(sample_dir, "NA12878_sub_R1.fastq")
    r2_out = os.path.join(sample_dir, "NA12878_sub_R2.fastq")
    
    # Generate a random seed
    seed = np.random.randint(10000)
    
    # Subsample R1
    cmd = ["seqtk", "sample", "-s", str(seed), Config.NA12878_FASTQ_R1, str(Config.SUBSAMPLE_RATIO)]
    with open(r1_out, "w") as f:
        subprocess.run(cmd, stdout=f, check=True)
    
    # Compress R1
    subprocess.run(["gzip", r1_out], check=True)
    
    # Subsample R2 with same seed
    cmd = ["seqtk", "sample", "-s", str(seed), Config.NA12878_FASTQ_R2, str(Config.SUBSAMPLE_RATIO)]
    with open(r2_out, "w") as f:
        subprocess.run(cmd, stdout=f, check=True)
    
    # Compress R2
    subprocess.run(["gzip", r2_out], check=True)
    
    return r1_out_gz, r2_out_gz

def run_alignment(fastq_r1, fastq_r2, output_bam, params):
    """Run BWA-MEM alignment with the best parameters from Stage I."""
    # Extract parameters
    seed_length = params["seed_length"]
    match_score = params["match_score"]
    mismatch_penalty = params["mismatch_penalty"]
    gap_open_penalty = params["gap_open_penalty"]
    gap_extension_penalty = params["gap_extension_penalty"]
    
    # Create temp directory for intermediate files
    temp_dir = os.path.join(Config.OUTPUT_DIR, "temp", f"align_{params['trial_number']}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Run BWA-MEM with best params from Stage I
    cmd = [
        Config.BWA, "mem",
        "-t", str(Config.THREADS),
        "-k", str(seed_length),
        "-A", str(match_score),
        "-B", str(mismatch_penalty),
        "-O", str(gap_open_penalty),
        "-E", str(gap_extension_penalty),
        "-R", f"@RG\\tID:NA12878\\tSM:NA12878\\tPL:ILLUMINA",
        Config.REFERENCE_GENOME,
        fastq_r1,
        fastq_r2
    ]
    
    # Pipe output to samtools for sorting
    raw_bam = os.path.join(temp_dir, "raw.bam")
    sort_cmd = [Config.SAMTOOLS, "sort", "-@", str(Config.THREADS), "-o", raw_bam]
    
    # Run the piped commands
    logger.info("Running alignment with best parameters from Stage I")
    
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(sort_cmd, stdin=p1.stdout, stdout=subprocess.PIPE)
    p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits
    p2.communicate()
    
    # Index the BAM
    run_command([Config.SAMTOOLS, "index", raw_bam], "Indexing BAM")
    
    # Mark duplicates
    dedup_bam = os.path.join(temp_dir, "dedup.bam")
    metrics_file = os.path.join(temp_dir, "metrics.txt")
    
    optical_distance = params["optical_duplicate_pixel_distance"]
    
    cmd = [
        Config.PICARD, "MarkDuplicates",
        f"I={raw_bam}",
        f"O={dedup_bam}",
        f"M={metrics_file}",
        f"OPTICAL_DUPLICATE_PIXEL_DISTANCE={optical_distance}",
        "CREATE_INDEX=true"
    ]
    run_command(cmd, "Marking duplicates")
    
    # Copy final BAM to output location
    shutil.copy2(dedup_bam, output_bam)
    shutil.copy2(dedup_bam.replace(".bam", ".bai"), output_bam.replace(".bam", ".bai"))
    
    # Clean up temp directory if successful
    if config["output"]["keep_intermediate"] == False:
        shutil.rmtree(temp_dir)
    
    return output_bam

def run_variant_calling(input_bam, output_vcf, trial_params):
    """Run GATK HaplotypeCaller with parameters from Optuna trial."""
    # Extract parameters
    heterozygosity = trial_params["heterozygosity"]
    indel_heterozygosity = trial_params["indel_heterozygosity"]
    pcr_indel_model = trial_params["pcr_indel_model"]
    min_pruning = trial_params["min_pruning"]
    standard_min_confidence_threshold = trial_params["standard_min_confidence_threshold"]
    
    # Build command with trial parameters
    cmd = [
        Config.GATK, "HaplotypeCaller",
        "-R", Config.REFERENCE_GENOME,
        "-I", input_bam,
        "-O", output_vcf,
        "--heterozygosity", str(heterozygosity),
        "--indel-heterozygosity", str(indel_heterozygosity),
        "--pcr-indel-model", pcr_indel_model,
        "--min-pruning", str(min_pruning),
        "--standard-min-confidence-threshold-for-calling", str(standard_min_confidence_threshold),
        "--native-pair-hmm-threads", str(Config.THREADS)
    ]
    
    # Optional parameters based on trial
    if "max_alternate_alleles" in trial_params:
        cmd.extend(["--max-alternate-alleles", str(trial_params["max_alternate_alleles"])])
    
    if "max_genotype_count" in trial_params:
        cmd.extend(["--max-genotype-count", str(trial_params["max_genotype_count"])])
    
    logger.info(f"Running HaplotypeCaller with parameters: heterozygosity={heterozygosity}, " +
                f"indel_heterozygosity={indel_heterozygosity}, pcr_indel_model={pcr_indel_model}, " +
                f"min_pruning={min_pruning}, standard_min_confidence_threshold={standard_min_confidence_threshold}")
    
    run_command(cmd, "Running HaplotypeCaller")
    
    # Index the VCF if needed
    if not output_vcf.endswith(".gz"):
        bgzip_cmd = [Config.BGZIP, output_vcf]
        run_command(bgzip_cmd, "Compressing VCF")
        output_vcf += ".gz"
    
    tabix_cmd = [Config.TABIX, "-p", "vcf", output_vcf]
    run_command(tabix_cmd, "Indexing VCF")
    
    return output_vcf

def evaluate_variants(vcf_file, trial_number):
    """Evaluate called variants against GIAB truth set using RTG Tools."""
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
        "--all-records"
    ]
    run_command(cmd, "Evaluating variants with RTG vcfeval")
    
    # Parse summary.txt to get F-measure, precision, and recall
    summary_file = os.path.join(eval_dir, "summary.txt")
    metrics = {}
    
    with open(summary_file, 'r') as f:
        for line in f:
            if line.startswith("F-measure"):
                metrics["f_measure"] = float(line.split()[-1])
            elif line.startswith("Precision"):
                metrics["precision"] = float(line.split()[-1])
            elif line.startswith("Recall"):
                metrics["recall"] = float(line.split()[-1])
    
    logger.info(f"Evaluation metrics - F-measure: {metrics.get('f_measure', 0)}, " +
                f"Precision: {metrics.get('precision', 0)}, Recall: {metrics.get('recall', 0)}")
    
    # Read from the weighted ROC file to get more detailed metrics
    roc_file = os.path.join(eval_dir, "weighted_roc.tsv")
    if os.path.exists(roc_file):
        try:
            roc_data = pd.read_csv(roc_file, sep='\t')
            # Get the max F-measure from the ROC curve
            if not roc_data.empty and 'F_measure' in roc_data.columns:
                best_f = roc_data['F_measure'].max()
                metrics["best_f_measure"] = best_f
                logger.info(f"Best F-measure from ROC curve: {best_f}")
        except Exception as e:
            logger.warning(f"Failed to read ROC data: {str(e)}")
    
    # If splitting by variant type, get separate metrics for SNPs and INDELs
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
    
    return metrics

def objective(trial):
    """Optuna objective function for HaplotypeCaller parameter optimization."""
    # Generate trial parameters
    params = {
        'trial_number': trial.number,
        'heterozygosity': trial.suggest_float('heterozygosity', 0.0005, 0.003),
        'indel_heterozygosity': trial.suggest_float('indel_heterozygosity', 0.00005, 0.0005),
        'pcr_indel_model': trial.suggest_categorical('pcr_indel_model', 
                                                  ['NONE', 'HOSTILE', 'AGGRESSIVE', 'CONSERVATIVE']),
        'min_pruning': trial.suggest_int('min_pruning', 1, 3),
        'standard_min_confidence_threshold': trial.suggest_int('standard_min_confidence_threshold', 10, 50)
    }
    
    # Optional parameters based on configuration
    if config["parameter_ranges"]["haplotype_caller"].get("max_alternate_alleles", {}).get("enabled", False):
        params['max_alternate_alleles'] = trial.suggest_int('max_alternate_alleles', 
                                                          config["parameter_ranges"]["haplotype_caller"]["max_alternate_alleles"]["min"],
                                                          config["parameter_ranges"]["haplotype_caller"]["max_alternate_alleles"]["max"])
    
    if config["parameter_ranges"]["haplotype_caller"].get("max_genotype_count", {}).get("enabled", False):
        params['max_genotype_count'] = trial.suggest_int('max_genotype_count',
                                                       config["parameter_ranges"]["haplotype_caller"]["max_genotype_count"]["min"],
                                                       config["parameter_ranges"]["haplotype_caller"]["max_genotype_count"]["max"])
    
    logger.info(f"Starting trial {trial.number} with parameters: {params}")
    
    # Output files for this trial
    trial_dir = os.path.join(Config.OUTPUT_DIR, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)
    
    # Get input BAM (prepared once and reused)
    input_bam = prepare_input_bam()
    
    # Output VCF for this trial
    output_vcf = os.path.join(trial_dir, "variants.vcf")
    
    try:
        # Time the variant calling pipeline
        start_time = time.time()
        
        # Run variant calling with trial parameters
        output_vcf = run_variant_calling(input_bam, output_vcf, params)
        
        # Evaluate the results
        metrics = evaluate_variants(output_vcf, trial.number)
        
        # Calculate runtime
        runtime = time.time() - start_time
        
        # Combine the metrics with specified weights
        primary_metric = config["evaluation"]["primary_metric"]
        combined_metric = metrics.get(primary_metric, 0)
        
        # Apply weights for multi-objective optimization if specified
        if primary_metric == "combined":
            w_snp = config["evaluation"]["weights"]["snp_f_measure"]
            w_indel = config["evaluation"]["weights"]["indel_f_measure"]
            
            snp_f = metrics.get("snp_f_measure", 0)
            indel_f = metrics.get("indel_f_measure", 0)
            
            combined_metric = (w_snp * snp_f + w_indel * indel_f) / (w_snp + w_indel)
        
        # Add runtime penalty if specified
        if config["evaluation"]["weights"]["runtime"] > 0:
            runtime_weight = config["evaluation"]["weights"]["runtime"]
            max_runtime = config["evaluation"]["max_runtime"]
            runtime_penalty = min(1.0, runtime / max_runtime) * runtime_weight
            combined_metric = combined_metric * (1 - runtime_penalty)
        
        # Log results
        logger.info(f"Trial {trial.number} completed - {primary_metric}: {combined_metric:.4f}, Runtime: {runtime:.2f}s")
        
        # Save trial details to CSV
        trial_results = {
            'trial_number': trial.number,
            'heterozygosity': params['heterozygosity'],
            'indel_heterozygosity': params['indel_heterozygosity'],
            'pcr_indel_model': params['pcr_indel_model'],
            'min_pruning': params['min_pruning'],
            'standard_min_confidence_threshold': params['standard_min_confidence_threshold'],
            'f_measure': metrics.get('f_measure', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'runtime': runtime
        }
        
        # Add optional parameters if they were used
        if 'max_alternate_alleles' in params:
            trial_results['max_alternate_alleles'] = params['max_alternate_alleles']
        
        if 'max_genotype_count' in params:
            trial_results['max_genotype_count'] = params['max_genotype_count']
        
        # Add variant-type specific metrics if available
        if 'snp_f_measure' in metrics:
            trial_results['snp_f_measure'] = metrics['snp_f_measure']
            trial_results['snp_precision'] = metrics.get('snp_precision', 0)
            trial_results['snp_recall'] = metrics.get('snp_recall', 0)
        
        if 'indel_f_measure' in metrics:
            trial_results['indel_f_measure'] = metrics['indel_f_measure']
            trial_results['indel_precision'] = metrics.get('indel_precision', 0)
            trial_results['indel_recall'] = metrics.get('indel_recall', 0)
        
        df = pd.DataFrame([trial_results])
        results_csv = os.path.join(Config.OUTPUT_DIR, "trial_results.csv")
        
        # Append to CSV or create a new one
        if os.path.exists(results_csv):
            df.to_csv(results_csv, mode='a', header=False, index=False)
        else:
            df.to_csv(results_csv, index=False)
        
        return combined_metric
    
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {str(e)}")
        return 0.0  # Return poorest score on error

def main():
    """Main function to run the optimization."""
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    logger.info("Starting GATK germline variant pipeline optimization - Stage II (HaplotypeCaller)")
    
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
        study_name="gatk_haplotypecaller_optimization",
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )
    
    # Run optimization
    study.optimize(objective, n_trials=Config.N_TRIALS, timeout=Config.TIMEOUT)
    
    # Report best parameters
    logger.info(f"Optimization completed. Best value: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    
    # Save best parameters as YAML
    best_params = {
        "haplotype_caller": {
            "heterozygosity": study.best_params["heterozygosity"],
            "indel_heterozygosity": study.best_params["indel_heterozygosity"],
            "pcr_indel_model": study.best_params["pcr_indel_model"],
            "min_pruning": study.best_params["min_pruning"],
            "standard_min_confidence_threshold": study.best_params["standard_min_confidence_threshold"]
        }
    }
    
    # Add optional parameters if they were optimized
    if "max_alternate_alleles" in study.best_params:
        best_params["haplotype_caller"]["max_alternate_alleles"] = study.best_params["max_alternate_alleles"]
    
    if "max_genotype_count" in study.best_params:
        best_params["haplotype_caller"]["max_genotype_count"] = study.best_params["max_genotype_count"]
    
    with open(os.path.join(Config.OUTPUT_DIR, "stage2_best_params.yaml"), 'w') as f:
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
        contour_params = [
            ("heterozygosity", "indel_heterozygosity"),
            ("heterozygosity", "min_pruning"),
            ("indel_heterozygosity", "pcr_indel_model")
        ]
        
        for param1, param2 in contour_params:
            if param1 in study.best_params and param2 in study.best_params:
                fig = plot_contour(study, params=[param1, param2])
                fig.write_image(os.path.join(Config.OUTPUT_DIR, f"contour_{param1}_vs_{param2}.png"))
    except Exception as e:
        logger.warning(f"Could not generate visualizations: {str(e)}")
    
    # Run final pipeline with best parameters on full dataset
    if config.get("run_full_dataset", False):
        logger.info("Running with best parameters on full dataset...")
        # Implementation for full dataset run would go here
    
    logger.info("Stage II optimization pipeline completed successfully")

if __name__ == "__main__":
    main()


