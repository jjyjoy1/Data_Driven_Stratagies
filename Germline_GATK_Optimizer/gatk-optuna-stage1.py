#!/usr/bin/env python3
# Stage I Configuration Loading Fix - Add to the beginning of your gatk-optuna-optimization.py file

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
import yaml
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gatk_optimization.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("GATK-Optuna")

# Load configuration
def load_config(config_path="stage1_config.yaml"):
    """Load configuration from YAML file."""
    logger.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        sys.exit(1)

# Load configuration
config = load_config()

# Configuration class populated from YAML
class Config:
    # Paths
    REFERENCE_GENOME = config["reference"]["genome"]
    NA12878_FASTQ_R1 = config["samples"]["NA12878"]["fastq_r1"]
    NA12878_FASTQ_R2 = config["samples"]["NA12878"]["fastq_r2"]
    GIAB_VCF = config["truth_sets"]["GIAB"]["vcf"]
    GIAB_BED = config["truth_sets"]["GIAB"]["bed"]
    OUTPUT_DIR = config["output"]["directory"]
    
    # Tools - assume they're in PATH unless specified in config
    BWA = config.get("tools", {}).get("bwa", "bwa")
    SAMTOOLS = config.get("tools", {}).get("samtools", "samtools")
    GATK = config.get("tools", {}).get("gatk", "gatk")
    PICARD = config.get("tools", {}).get("picard", "picard")
    BGZIP = config.get("tools", {}).get("bgzip", "bgzip")
    TABIX = config.get("tools", {}).get("tabix", "tabix")
    RTGTOOLS = config.get("tools", {}).get("rtg", "rtg")
    
    # Number of CPU cores to use
    THREADS = config["computing"]["threads"]
    
    # Subsampling percentage (0-1)
    SUBSAMPLE_RATIO = config["subsampling"]["ratio"]
    
    # RTG Tools evaluation settings
    RTG_SDF_DIR = config["truth_sets"]["GIAB"]["rtg_sdf"]
    
    # Optimization settings
    N_TRIALS = config["optuna"]["n_trials"]
    TIMEOUT = config["optuna"]["timeout"]
    
    # Parameter ranges
    BWA_SEED_LENGTH_MIN = config["parameter_ranges"]["bwa_mem"]["seed_length"]["min"]
    BWA_SEED_LENGTH_MAX = config["parameter_ranges"]["bwa_mem"]["seed_length"]["max"]
    BWA_MATCH_SCORE_MIN = config["parameter_ranges"]["bwa_mem"]["match_score"]["min"]
    BWA_MATCH_SCORE_MAX = config["parameter_ranges"]["bwa_mem"]["match_score"]["max"]
    BWA_MISMATCH_PENALTY_MIN = config["parameter_ranges"]["bwa_mem"]["mismatch_penalty"]["min"]
    BWA_MISMATCH_PENALTY_MAX = config["parameter_ranges"]["bwa_mem"]["mismatch_penalty"]["max"]
    BWA_GAP_OPEN_PENALTY_MIN = config["parameter_ranges"]["bwa_mem"]["gap_open_penalty"]["min"]
    BWA_GAP_OPEN_PENALTY_MAX = config["parameter_ranges"]["bwa_mem"]["gap_open_penalty"]["max"]
    BWA_GAP_EXTENSION_PENALTY_MIN = config["parameter_ranges"]["bwa_mem"]["gap_extension_penalty"]["min"]
    BWA_GAP_EXTENSION_PENALTY_MAX = config["parameter_ranges"]["bwa_mem"]["gap_extension_penalty"]["max"]
    OPTICAL_DUPLICATE_PIXEL_DISTANCE_MIN = config["parameter_ranges"]["mark_duplicates"]["optical_duplicate_pixel_distance"]["min"]
    OPTICAL_DUPLICATE_PIXEL_DISTANCE_MAX = config["parameter_ranges"]["mark_duplicates"]["optical_duplicate_pixel_distance"]["max"]

# Display configuration summary
logger.info("Configuration Summary:")
logger.info(f"Reference Genome: {Config.REFERENCE_GENOME}")
logger.info(f"FASTQ Files: {Config.NA12878_FASTQ_R1}, {Config.NA12878_FASTQ_R2}")
logger.info(f"Truth Set: {Config.GIAB_VCF}")
logger.info(f"Output Directory: {Config.OUTPUT_DIR}")
logger.info(f"Threads: {Config.THREADS}")
logger.info(f"Subsample Ratio: {Config.SUBSAMPLE_RATIO}")
logger.info(f"Number of Trials: {Config.N_TRIALS}")

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

def subsample_fastq(input_fastq, output_fastq, ratio):
    """Subsample FASTQ file using seqtk."""
    n_reads = int(subprocess.check_output(f"zcat {input_fastq} | wc -l", shell=True).decode().strip()) // 4
    n_sample = int(n_reads * ratio)
    
    # Generate a random seed
    seed = np.random.randint(10000)
    
    cmd = ["seqtk", "sample", "-s", str(seed), input_fastq, str(n_sample)]
    with open(output_fastq, "w") as f:
        subprocess.run(cmd, stdout=f, check=True)
    
    # Compress the output
    subprocess.run(["gzip", output_fastq], check=True)
    return output_fastq + ".gz"

def prepare_subsampled_data():
    """Prepare subsampled FASTQ files for NA12878."""
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    sample_dir = os.path.join(Config.OUTPUT_DIR, "subsampled")
    os.makedirs(sample_dir, exist_ok=True)
    
    r1_out = os.path.join(sample_dir, "NA12878_sub_R1.fastq")
    r2_out = os.path.join(sample_dir, "NA12878_sub_R2.fastq")
    
    r1_out_gz = subsample_fastq(Config.NA12878_FASTQ_R1, r1_out, Config.SUBSAMPLE_RATIO)
    r2_out_gz = subsample_fastq(Config.NA12878_FASTQ_R2, r2_out, Config.SUBSAMPLE_RATIO)
    
    return r1_out_gz, r2_out_gz

def run_alignment(fastq_r1, fastq_r2, output_bam, trial_params):
    """Run BWA-MEM alignment with the parameters from Optuna trial."""
    # Extract parameters
    seed_length = trial_params["seed_length"]
    match_score = trial_params["match_score"]
    mismatch_penalty = trial_params["mismatch_penalty"]
    gap_open_penalty = trial_params["gap_open_penalty"]
    gap_extension_penalty = trial_params["gap_extension_penalty"]
    
    # Create temp directory for intermediate files
    temp_dir = os.path.join(Config.OUTPUT_DIR, "temp", f"trial_{trial_params['trial_number']}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Run BWA-MEM with params
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
    logger.info(f"Running alignment with parameters: k={seed_length}, A={match_score}, B={mismatch_penalty}, O={gap_open_penalty}, E={gap_extension_penalty}")
    
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(sort_cmd, stdin=p1.stdout, stdout=subprocess.PIPE)
    p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits
    p2.communicate()
    
    # Index the BAM
    run_command([Config.SAMTOOLS, "index", raw_bam], "Indexing BAM")
    
    # Mark duplicates
    dedup_bam = os.path.join(temp_dir, "dedup.bam")
    metrics_file = os.path.join(temp_dir, "metrics.txt")
    
    optical_distance = trial_params["optical_duplicate_pixel_distance"]
    
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
    shutil.rmtree(temp_dir)
    
    return output_bam

def run_variant_calling(input_bam, output_vcf):
    """Run GATK HaplotypeCaller with default parameters."""
    cmd = [
        Config.GATK, "HaplotypeCaller",
        "-R", Config.REFERENCE_GENOME,
        "-I", input_bam,
        "-O", output_vcf,
        "--native-pair-hmm-threads", str(Config.THREADS)
    ]
    run_command(cmd, "Running HaplotypeCaller")
    
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
    
    # Parse summary.txt to get F-measure
    summary_file = os.path.join(eval_dir, "summary.txt")
    with open(summary_file, 'r') as f:
        for line in f:
            if line.startswith("F-measure"):
                f_measure = float(line.split()[-1])
                return f_measure
    
    # If parsing failed, return a poor score
    return 0.0

def objective(trial):
    """Optuna objective function for alignment parameter optimization."""
    # Generate trial parameters using ranges from config
    params = {
        'trial_number': trial.number,
        'seed_length': trial.suggest_int('seed_length', Config.BWA_SEED_LENGTH_MIN, Config.BWA_SEED_LENGTH_MAX),
        'match_score': trial.suggest_int('match_score', Config.BWA_MATCH_SCORE_MIN, Config.BWA_MATCH_SCORE_MAX),
        'mismatch_penalty': trial.suggest_int('mismatch_penalty', Config.BWA_MISMATCH_PENALTY_MIN, Config.BWA_MISMATCH_PENALTY_MAX),
        'gap_open_penalty': trial.suggest_int('gap_open_penalty', Config.BWA_GAP_OPEN_PENALTY_MIN, Config.BWA_GAP_OPEN_PENALTY_MAX),
        'gap_extension_penalty': trial.suggest_int('gap_extension_penalty', Config.BWA_GAP_EXTENSION_PENALTY_MIN, Config.BWA_GAP_EXTENSION_PENALTY_MAX),
        'optical_duplicate_pixel_distance': trial.suggest_int('optical_duplicate_pixel_distance', Config.OPTICAL_DUPLICATE_PIXEL_DISTANCE_MIN, Config.OPTICAL_DUPLICATE_PIXEL_DISTANCE_MAX)
    }
    
    logger.info(f"Starting trial {trial.number} with parameters: {params}")
    
    # Output files for this trial
    trial_dir = os.path.join(Config.OUTPUT_DIR, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)
    
    output_bam = os.path.join(trial_dir, "aligned.bam")
    output_vcf = os.path.join(trial_dir, "variants.vcf.gz")
    
    try:
        # Time the entire pipeline
        start_time = time.time()
        
        # Run alignment with trial parameters
        run_alignment(fastq_r1, fastq_r2, output_bam, params)
        
        # Run variant calling with default parameters
        run_variant_calling(output_bam, output_vcf)
        
        # Evaluate the results
        f_measure = evaluate_variants(output_vcf, trial.number)
        
        # Calculate runtime
        runtime = time.time() - start_time
        
        # Log results
        logger.info(f"Trial {trial.number} completed - F-measure: {f_measure:.4f}, Runtime: {runtime:.2f}s")
        
        # Save trial details to CSV
        trial_results = {
            'trial_number': trial.number,
            'seed_length': params['seed_length'],
            'match_score': params['match_score'],
            'mismatch_penalty': params['mismatch_penalty'],
            'gap_open_penalty': params['gap_open_penalty'],
            'gap_extension_penalty': params['gap_extension_penalty'],
            'optical_duplicate_pixel_distance': params['optical_duplicate_pixel_distance'],
            'f_measure': f_measure,
            'runtime': runtime
        }
        
        df = pd.DataFrame([trial_results])
        results_csv = os.path.join(Config.OUTPUT_DIR, "trial_results.csv")
        
        # Append to CSV or create a new one
        if os.path.exists(results_csv):
            df.to_csv(results_csv, mode='a', header=False, index=False)
        else:
            df.to_csv(results_csv, index=False)
        
        return f_measure
    
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {str(e)}")
        return 0.0  # Return poorest score on error

def main():
    """Main function to run the optimization."""
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    logger.info("Starting GATK germline variant pipeline optimization")
    
    # Prepare subsampled data
    global fastq_r1, fastq_r2
    logger.info(f"Preparing subsampled data at {Config.SUBSAMPLE_RATIO*100}%")
    fastq_r1, fastq_r2 = prepare_subsampled_data()
    
    # Create Optuna study
    logger.info(f"Creating Optuna study with {Config.N_TRIALS} trials")
    study = optuna.create_study(
        study_name="gatk_alignment_optimization",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=Config.N_TRIALS, timeout=Config.TIMEOUT)
    
    # Report best parameters
    logger.info(f"Optimization completed. Best F-measure: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    
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
        
        # Plot contour plots
        for param1 in study.best_params:
            for param2 in study.best_params:
                if param1 != param2:
                    fig = plot_contour(study, params=[param1, param2])
                    fig.write_image(os.path.join(Config.OUTPUT_DIR, f"contour_{param1}_vs_{param2}.png"))
    except Exception as e:
        logger.warning(f"Could not generate visualizations: {str(e)}")
    
    # Apply best parameters to full dataset
    logger.info("Running with best parameters on full dataset...")
    # (Implementation for full dataset run would go here)
    
    logger.info("Optimization pipeline completed successfully")

if __name__ == "__main__":
    main()
    
