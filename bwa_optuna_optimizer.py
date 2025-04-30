import optuna
import subprocess
import os
import uuid
import shutil
import re

def objective(trial):
    # Parameters to optimize for BWA MEM
    min_seed_len = trial.suggest_int('min_seed_len', 15, 30)  # -k parameter
    band_width = trial.suggest_int('band_width', 50, 200)  # -w parameter
    z_dropoff = trial.suggest_int('z_dropoff', 50, 150)  # -d parameter
    seed_split_ratio = trial.suggest_float('seed_split_ratio', 0.5, 2.0)  # -r parameter
    max_occ = trial.suggest_int('max_occ', 5, 100)  # -c parameter
    match_score = trial.suggest_int('match_score', 1, 5)  # -A parameter
    mismatch_penalty = trial.suggest_int('mismatch_penalty', 2, 10)  # -B parameter
    gap_open_penalty = trial.suggest_int('gap_open_penalty', 2, 15)  # -O parameter
    gap_ext_penalty = trial.suggest_int('gap_ext_penalty', 1, 5)  # -E parameter
    
    # Run BWA MEM with these parameters and evaluate performance
    performance_metric = run_bwa_mem_and_evaluate(
        min_seed_len, band_width, z_dropoff, seed_split_ratio, max_occ,
        match_score, mismatch_penalty, gap_open_penalty, gap_ext_penalty
    )
    
    return performance_metric

def run_bwa_mem_and_evaluate(min_seed_len, band_width, z_dropoff, seed_split_ratio, 
                            max_occ, match_score, mismatch_penalty, 
                            gap_open_penalty, gap_ext_penalty):
    # Create a unique working directory
    work_dir = f"./bwa_optimize_{uuid.uuid4()}"
    os.makedirs(work_dir, exist_ok=True)
    
    try:
        # Paths to your reference, reads, and truth set
        reference = "/path/to/your/reference.fa"
        reads_1 = "/path/to/your/reads_1.fq"
        reads_2 = "/path/to/your/reads_2.fq"  # For paired-end reads
        
        # Construct BWA MEM command with parameters
        bwa_cmd = (
            f"bwa mem -t 8 "
            f"-k {min_seed_len} "
            f"-w {band_width} "
            f"-d {z_dropoff} "
            f"-r {seed_split_ratio} "
            f"-c {max_occ} "
            f"-A {match_score} "
            f"-B {mismatch_penalty} "
            f"-O {gap_open_penalty} "
            f"-E {gap_ext_penalty} "
            f"{reference} {reads_1} {reads_2} > {work_dir}/aligned.sam"
        )
        
        # Run BWA MEM
        subprocess.run(bwa_cmd, shell=True, check=True)
        
        # Convert to BAM, sort and index
        subprocess.run(f"samtools view -Sb {work_dir}/aligned.sam > {work_dir}/aligned.bam", 
                       shell=True, check=True)
        subprocess.run(f"samtools sort {work_dir}/aligned.bam -o {work_dir}/sorted.bam", 
                       shell=True, check=True)
        subprocess.run(f"samtools index {work_dir}/sorted.bam", 
                       shell=True, check=True)
        
        # Evaluate alignment quality
        # Here you'll need to define your evaluation metric
        # Some options:
        # 1. If you have a truth BAM, you can use tools like samtools flagstat or Picard
        # 2. For NGS data, you could run a variant caller and check concordance with known variants
        # 3. Calculate mapping rate, properly paired rate, etc.
        
        # Example: Get mapping statistics
        flagstat_cmd = f"samtools flagstat {work_dir}/sorted.bam > {work_dir}/flagstat.txt"
        subprocess.run(flagstat_cmd, shell=True, check=True)
        
        # Parse the flagstat output to get metrics
        with open(f"{work_dir}/flagstat.txt", 'r') as f:
            flagstat_data = f.read()
            
        # Extract mapping rate
        mapping_rate_match = re.search(r'(\d+\.\d+)% mapped', flagstat_data)
        if mapping_rate_match:
            mapping_rate = float(mapping_rate_match.group(1))
        else:
            mapping_rate = 0
            
        # Extract properly paired rate (for paired-end data)
        proper_pair_match = re.search(r'(\d+\.\d+)% properly paired', flagstat_data)
        if proper_pair_match:
            proper_pair_rate = float(proper_pair_match.group(1))
        else:
            proper_pair_rate = 0
        
        # You can define a combined metric based on what's important for your use case
        # For example: 0.7 * mapping_rate + 0.3 * proper_pair_rate
        performance_metric = 0.7 * mapping_rate + 0.3 * proper_pair_rate
        
        return performance_metric
        
    except Exception as e:
        print(f"Error running BWA MEM with parameters: {e}")
        return 0  # Return worst possible score on failure
        
    finally:
        # Clean up
        shutil.rmtree(work_dir)

# Create the study and run optimization
study = optuna.create_study(
    study_name="bwa_mem_optimization",
    storage="sqlite:///bwa_optimization.db",  # Save results in a database
    direction="maximize",  # We want to maximize our performance metric
    load_if_exists=True  # Resume if the study already exists
)

# Run 50 trials
study.optimize(objective, n_trials=50)

# Print results
print("Best parameters:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")
print(f"Best performance: {study.best_value}")

# Visualize the optimization process (if running in a notebook or with display)
try:
    from optuna.visualization import plot_optimization_history, plot_param_importances
    
    # Plot optimization history
    fig = plot_optimization_history(study)
    fig.write_image("bwa_optimization_history.png")
    
    # Plot parameter importances
    fig = plot_param_importances(study)
    fig.write_image("bwa_parameter_importances.png")
except:
    print("Visualization couldn't be generated. Make sure you have plotly installed.")



