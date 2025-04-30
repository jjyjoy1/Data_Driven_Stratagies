#!/usr/bin/env python3
# GATK Germline Variant Pipeline Optimization with Optuna (Stage I)
# Focus on alignment parameters optimization using direct alignment metrics

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
import re
import json
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gatk_stage1_optimization.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("GATK-Optuna-Stage1")

# Load configuration
def load_config(config_path="stage1_config_v2.yaml"):
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
    OUTPUT_DIR = os.path.join(config["output"]["directory"], "stage1")
    
    # Tools - assume they're in PATH unless specified in config
    BWA = config.get("tools", {}).get("bwa", "bwa")
    SAMTOOLS = config.get("tools", {}).get("samtools", "samtools")
    GATK = config.get("tools", {}).get("gatk", "gatk")
    PICARD = config.get("tools", {}).get("picard", "picard")
    SEQTK = config.get("tools", {}).get("seqtk", "seqtk")
    
    # Number of CPU cores to use
    THREADS = config["computing"]["threads"]
    
    # Subsampling percentage (0-1)
    SUBSAMPLE_RATIO = config["subsampling"]["ratio"]
    
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
    
    # Alignment metrics evaluation weights
    METRICS_WEIGHTS = config.get("alignment_metrics", {}).get("weights", {
        "mapped_pct": 0.25,
        "properly_paired_pct": 0.25,
        "avg_quality": 0.15,
        "error_rate": -0.20,  # Negative weight since lower is better
        "duplicate_pct": -0.10,  # Negative weight since lower is better
        "soft_clip_pct": -0.05  # Negative weight since lower is better
    })

# Display configuration summary
logger.info("Configuration Summary:")
logger.info(f"Reference Genome: {Config.REFERENCE_GENOME}")
logger.info(f"FASTQ Files: {Config.NA12878_FASTQ_R1}, {Config.NA12878_FASTQ_R2}")
logger.info(f"Output Directory: {Config.OUTPUT_DIR}")
logger.info(f"Threads: {Config.THREADS}")
logger.info(f"Subsample Ratio: {Config.SUBSAMPLE_RATIO}")
logger.info(f"Number of Trials: {Config.N_TRIALS}")
logger.info(f"Alignment Metrics Weights: {Config.METRICS_WEIGHTS}")

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
    
    cmd = [Config.SEQTK, "sample", "-s", str(seed), input_fastq, str(n_sample)]
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
    if not config.get("output", {}).get("keep_intermediate", False):
        shutil.rmtree(temp_dir)
    
    return output_bam, metrics_file

def evaluate_alignment_quality(bam_file, duplicates_metrics_file):
    """Evaluate alignment quality metrics directly from BAM file."""
    metrics = {}
    
    # Run samtools flagstat
    flagstat_cmd = [Config.SAMTOOLS, "flagstat", bam_file]
    flagstat_output, _, _ = run_command(flagstat_cmd, "Getting alignment statistics")
    
    # Parse flagstat output
    for line in flagstat_output.split('\n'):
        if "properly paired" in line and "%" in line:
            properly_paired_pct = float(line.split('(')[1].split('%')[0])
            metrics["properly_paired_pct"] = properly_paired_pct
        elif "mapped (" in line and "%" in line:
            mapped_pct = float(line.split('(')[1].split('%')[0])
            metrics["mapped_pct"] = mapped_pct
        elif "duplicates" in line and "%" in line:
            duplicate_pct = float(line.split('(')[1].split('%')[0])
            metrics["duplicate_pct"] = duplicate_pct
    
    # Run samtools stats for more detailed metrics
    stats_cmd = [Config.SAMTOOLS, "stats", bam_file]
    stats_output, _, _ = run_command(stats_cmd, "Getting detailed alignment statistics")
    
    # Parse stats output for error rate and quality
    for line in stats_output.split('\n'):
        if line.startswith('SN\terror rate:'):
            metrics["error_rate"] = float(line.split('\t')[2])
        elif line.startswith('SN\taverage quality:'):
            metrics["avg_quality"] = float(line.split('\t')[2])
    
    # Run samtools stats with base statistics for clip rate
    stats_cmd = [Config.SAMTOOLS, "stats", "-c", "1,30000,5", bam_file]
    stats_output, _, _ = run_command(stats_cmd, "Getting base-level statistics")
    
    # Calculate soft clipping percentage
    total_bases = 0
    soft_clipped_bases = 0
    
    for line in stats_output.split('\n'):
        if line.startswith('RL'):
            fields = line.split('\t')
            if len(fields) >= 3:
                total_bases += int(fields[1]) * int(fields[2])
        elif line.startswith('BC'):
            fields = line.split('\t')
            if len(fields) >= 3 and fields[1] == 'S':
                soft_clipped_bases += int(fields[2])
    
    if total_bases > 0:
        metrics["soft_clip_pct"] = (soft_clipped_bases / total_bases) * 100
    else:
        metrics["soft_clip_pct"] = 0
    
    # Get more accurate duplication rate from Picard metrics file
    try:
        with open(duplicates_metrics_file, 'r') as f:
            for line in f:
                if line.startswith('LIBRARY'):
                    header = line.strip().split('\t')
                    values = next(f).strip().split('\t')
                    metrics_dict = dict(zip(header, values))
                    if 'PERCENT_DUPLICATION' in metrics_dict:
                        metrics["picard_duplicate_pct"] = float(metrics_dict['PERCENT_DUPLICATION']) * 100
                    break
    except Exception as e:
        logger.warning(f"Could not parse Picard duplicate metrics: {str(e)}")
    
    # Calculate insert size statistics
    insert_cmd = [
        Config.PICARD, "CollectInsertSizeMetrics",
        f"I={bam_file}",
        "O=/dev/null",  # We don't need to save the output file
        "H=/dev/null",  # We don't need the histogram
        "VALIDATION_STRINGENCY=LENIENT"  # Be lenient about validation
    ]
    insert_output, insert_err, _ = run_command(insert_cmd, "Collecting insert size metrics", check=False)
    
    # Parse insert size metrics from stderr (Picard outputs to stderr)
    insert_median = None
    insert_mad = None
    
    if insert_err:
        for line in insert_err.split('\n'):
            if "MEDIAN_INSERT_SIZE" in line and "MEDIAN_ABSOLUTE_DEVIATION" in line:
                header = line.strip().split('\t')
                values = next(insert_err.split('\n')[insert_err.split('\n').index(line) + 1], "").strip().split('\t')
                if len(values) >= len(header):
                    metrics_dict = dict(zip(header, values))
                    if 'MEDIAN_INSERT_SIZE' in metrics_dict:
                        try:
                            insert_median = float(metrics_dict['MEDIAN_INSERT_SIZE'])
                            metrics["median_insert_size"] = insert_median
                        except:
                            pass
                    if 'MEDIAN_ABSOLUTE_DEVIATION' in metrics_dict:
                        try:
                            insert_mad = float(metrics_dict['MEDIAN_ABSOLUTE_DEVIATION'])
                            metrics["insert_size_mad"] = insert_mad
                        except:
                            pass
    
    # Calculate overall alignment score based on weighted metrics
    alignment_score = 0
    
    for metric, weight in Config.METRICS_WEIGHTS.items():
        if metric in metrics:
            # Apply weight to metric
            alignment_score += metrics[metric] * weight
            logger.debug(f"Metric {metric}: {metrics[metric]:.2f} × {weight} = {metrics[metric] * weight:.2f}")
    
    metrics["alignment_score"] = alignment_score
    
    # Log all collected metrics
    logger.info(f"Alignment Metrics: mapped={metrics.get('mapped_pct', 0):.1f}%, " +
               f"properly_paired={metrics.get('properly_paired_pct', 0):.1f}%, " +
               f"duplicate={metrics.get('duplicate_pct', 0):.1f}%, " +
               f"error_rate={metrics.get('error_rate', 0) * 100:.3f}%, " +
               f"avg_quality={metrics.get('avg_quality', 0):.1f}, " +
               f"soft_clip={metrics.get('soft_clip_pct', 0):.1f}%, " +
               f"score={alignment_score:.2f}")
    
    return metrics, alignment_score

def run_qualimap(bam_file, output_dir):
    """Run Qualimap for additional alignment quality metrics."""
    try:
        qualimap_dir = os.path.join(output_dir, "qualimap")
        os.makedirs(qualimap_dir, exist_ok=True)
        
        cmd = [
            "qualimap", "bamqc",
            "-bam", bam_file,
            "-outdir", qualimap_dir,
            "-nt", str(Config.THREADS),
            "--java-mem-size=4G"
        ]
        
        run_command(cmd, "Running Qualimap for alignment QC", check=False)
        return qualimap_dir
    except Exception as e:
        logger.warning(f"Could not run Qualimap: {str(e)}")
        return None

def objective(trial):
    """Optuna objective function for alignment parameter optimization."""
    # Generate trial parameters
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
    
    try:
        # Time the entire pipeline
        start_time = time.time()
        
        # Run alignment with trial parameters
        output_bam, duplicates_metrics_file = run_alignment(fastq_r1, fastq_r2, output_bam, params)
        
        # Evaluate alignment quality directly
        metrics, alignment_score = evaluate_alignment_quality(output_bam, duplicates_metrics_file)
        
        # Run optional Qualimap analysis for additional metrics
        if config.get("alignment_metrics", {}).get("run_qualimap", False):
            qualimap_dir = run_qualimap(output_bam, trial_dir)
            if qualimap_dir:
                logger.info(f"Qualimap analysis completed: {qualimap_dir}")
        
        # Calculate runtime
        runtime = time.time() - start_time
        metrics["runtime"] = runtime
        
        # Log results
        logger.info(f"Trial {trial.number} completed - Alignment score: {alignment_score:.4f}, Runtime: {runtime:.2f}s")
        
        # Save trial details to CSV
        trial_results = {
            'trial_number': trial.number,
            'seed_length': params['seed_length'],
            'match_score': params['match_score'],
            'mismatch_penalty': params['mismatch_penalty'],
            'gap_open_penalty': params['gap_open_penalty'],
            'gap_extension_penalty': params['gap_extension_penalty'],
            'optical_duplicate_pixel_distance': params['optical_duplicate_pixel_distance'],
            'alignment_score': alignment_score,
            'runtime': runtime
        }
        
        # Add all metrics to results
        for key, value in metrics.items():
            if key not in trial_results:
                trial_results[key] = value
        
        df = pd.DataFrame([trial_results])
        results_csv = os.path.join(Config.OUTPUT_DIR, "trial_results.csv")
        
        # Append to CSV or create a new one
        if os.path.exists(results_csv):
            df.to_csv(results_csv, mode='a', header=False, index=False)
        else:
            df.to_csv(results_csv, index=False)
        
        return alignment_score
    
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {str(e)}")
        return 0.0  # Return poorest score on error

def main():
    """Main function to run the optimization."""
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    logger.info("Starting GATK germline variant pipeline optimization - Stage I (Alignment Parameters)")
    
    # Prepare subsampled data
    global fastq_r1, fastq_r2
    logger.info(f"Preparing subsampled data at {Config.SUBSAMPLE_RATIO*100}%")
    fastq_r1, fastq_r2 = prepare_subsampled_data()
    
    # Create Optuna study
    logger.info(f"Creating Optuna study with {Config.N_TRIALS} trials")
    
    # Set up pruner if enabled
    pruner = None
    if config["optuna"].get("pruner", None) == "MedianPruner":
        pruner = optuna.pruners.MedianPruner()
    elif config["optuna"].get("pruner", None) == "PercentilePruner":
        pruner = optuna.pruners.PercentilePruner(25.0)
    
    # Set up sampler
    sampler = None
    if config["optuna"].get("sampler", None) == "TPESampler":
        sampler = optuna.samplers.TPESampler(seed=config["optuna"].get("seed", 42))
    elif config["optuna"].get("sampler", None) == "RandomSampler":
        sampler = optuna.samplers.RandomSampler(seed=config["optuna"].get("seed", 42))
    elif config["optuna"].get("sampler", None) == "CmaEsSampler":
        sampler = optuna.samplers.CmaEsSampler(seed=config["optuna"].get("seed", 42))
    
    study = optuna.create_study(
        study_name="gatk_alignment_optimization",
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )
    
    # Run optimization
    study.optimize(objective, n_trials=Config.N_TRIALS, timeout=Config.TIMEOUT)
    
    # Report best parameters
    logger.info(f"Optimization completed. Best alignment score: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    
    # Save best parameters to YAML file
    best_params = {
        "bwa_mem": {
            "seed_length": study.best_params["seed_length"],
            "match_score": study.best_params["match_score"],
            "mismatch_penalty": study.best_params["mismatch_penalty"],
            "gap_open_penalty": study.best_params["gap_open_penalty"],
            "gap_extension_penalty": study.best_params["gap_extension_penalty"]
        },
        "mark_duplicates": {
            "optical_duplicate_pixel_distance": study.best_params["optical_duplicate_pixel_distance"]
        }
    }
    
    with open(os.path.join(Config.OUTPUT_DIR, "stage1_best_params.yaml"), 'w') as f:
        yaml.dump(best_params, f, default_flow_style=False)
    
    # Save study
    import joblib
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
    
    # Create a report
    logger.info("Creating alignment optimization report")
    
    # Get top 5 trials
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)[:5]
    
    report = f"""# Alignment Parameter Optimization Report

## Best Parameters

The best alignment parameters found after {len(study.trials)} trials:

- Seed length (`-k`): {study.best_params["seed_length"]}
- Match score (`-A`): {study.best_params["match_score"]}
- Mismatch penalty (`-B`): {study.best_params["mismatch_penalty"]}
- Gap open penalty (`-O`): {study.best_params["gap_open_penalty"]}
- Gap extension penalty (`-E`): {study.best_params["gap_extension_penalty"]}
- Optical duplicate pixel distance: {study.best_params["optical_duplicate_pixel_distance"]}

Alignment score: {study.best_value:.4f}

## Top 5 Parameter Combinations

| Rank | Seed Length | Match Score | Mismatch Penalty | Gap Open | Gap Extension | Optical Duplicate | Alignment Score |
|------|-------------|-------------|------------------|----------|---------------|-------------------|-----------------|
"""
    
    for i, trial in enumerate(top_trials):
        if trial.value is not None:
            report += f"| {i+1} | {trial.params.get('seed_length', 'N/A')} | {trial.params.get('match_score', 'N/A')} | {trial.params.get('mismatch_penalty', 'N/A')} | {trial.params.get('gap_open_penalty', 'N/A')} | {trial.params.get('gap_extension_penalty', 'N/A')} | {trial.params.get('optical_duplicate_pixel_distance', 'N/A')} | {trial.value:.4f} |\n"
    
    report += f"""
## Parameter Importance

The most important parameters for alignment quality (based on Optuna's feature importance):

"""
    
    # Calculate parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        for param, score in importance.items():
            report += f"- {param}: {score:.4f}\n"
    except:
        report += "Parameter importance calculation failed.\n"
    
    report += f"""
## Next Steps

1. The best parameters have been saved to `stage1_best_params.yaml`
2. Use these parameters for Stage II (HaplotypeCaller optimization)
3. For a full analysis of the results, see the CSV file with all trial metrics

Report generated on {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    with open(os.path.join(Config.OUTPUT_DIR, "alignment_optimization_report.md"), 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to {os.path.join(Config.OUTPUT_DIR, 'alignment_optimization_report.md')}")
    logger.info("Stage I optimization completed successfully")

if __name__ == "__main__":
    main()


