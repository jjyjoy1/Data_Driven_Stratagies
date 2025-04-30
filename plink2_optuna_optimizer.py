import optuna
import subprocess
import os
import re
import pandas as pd
import numpy as np
import uuid
import shutil
from scipy.stats import chi2

def objective(trial):
    # Create a dictionary to hold all parameters
    params = {}
    
    # 1. MAF/HWE filters
    params['maf'] = trial.suggest_float('maf', 0.001, 0.05)
    params['hwe'] = trial.suggest_float('hwe', 1e-10, 1e-4, log=True)
    
    # 2. Missingness filters
    params['geno'] = trial.suggest_float('geno', 0.01, 0.1)
    params['mind'] = trial.suggest_float('mind', 0.01, 0.1)
    
    # 3. Covariate selection
    # Assuming your covariate file contains these columns
    available_covars = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'age', 'sex']
    
    # Decide how many PCs to include (1-10)
    params['n_pcs'] = trial.suggest_int('n_pcs', 1, 10)
    pc_covars = [f'PC{i+1}' for i in range(params['n_pcs'])]
    
    # Decide whether to include age and sex
    params['include_age'] = trial.suggest_categorical('include_age', [True, False])
    params['include_sex'] = trial.suggest_categorical('include_sex', [True, False])
    
    selected_covars = pc_covars.copy()
    if params['include_age']:
        selected_covars.append('age')
    if params['include_sex']:
        selected_covars.append('sex')
    params['selected_covars'] = selected_covars
    
    # 4. Model parameters
    # Decide on the model type based on phenotype
    params['model_type'] = trial.suggest_categorical('model_type', ['linear', 'logistic'])
    
    # Decide on Firth correction approach (for logistic regression)
    if params['model_type'] == 'logistic':
        params['firth_approach'] = trial.suggest_categorical('firth_approach', 
                                                         ['none', 'firth', 'firth-fallback'])
    else:
        params['firth_approach'] = 'none'
    
    # 5. VIF/max-corr for multicollinearity
    params['use_vif'] = trial.suggest_categorical('use_vif', [True, False])
    if params['use_vif']:
        params['vif_threshold'] = trial.suggest_float('vif_threshold', 2.0, 50.0)
        params['max_corr'] = None
    else:
        params['vif_threshold'] = None
        params['max_corr'] = trial.suggest_float('max_corr', 0.6, 0.99)
    
    # 6. NEW: Kinship/relatedness adjustments
    params['use_king'] = trial.suggest_categorical('use_king', [True, False])
    if params['use_king']:
        params['king_cutoff'] = trial.suggest_float('king_cutoff', 0.0442, 0.177)  # 0.0442 = 3rd degree, 0.177 = 2nd degree
        
    # 7. NEW: Multiple testing correction
    params['adjust_method'] = trial.suggest_categorical('adjust_method', 
                                                     ['none', 'bonferroni', 'holm', 'sidak', 'fdr'])
    
    # 8. NEW: Genotype uncertainty parameters
    params['use_dosage'] = trial.suggest_categorical('use_dosage', [True, False])
    if params['use_dosage']:
        params['dosage_certainty'] = trial.suggest_float('dosage_certainty', 0.1, 1.0)
    
    # Run PLINK2 with these parameters and evaluate performance
    performance_metric = run_plink2_gwas(params)
    
    return performance_metric

def run_plink2_gwas(params):
    # Create a unique working directory
    work_dir = f"./plink2_optimize_{uuid.uuid4()}"
    os.makedirs(work_dir, exist_ok=True)
    
    try:
        # Paths to your data - Adjust these paths
        bed_file = "/path/to/your/data.bed"
        bim_file = "/path/to/your/data.bim"
        fam_file = "/path/to/your/data.fam"
        covar_file = "/path/to/your/covariates.txt"
        pheno_file = "/path/to/your/phenotype.txt"
        
        # For dosage data if used
        dosage_file = "/path/to/your/dosage.gz"  # Update with your dosage file path
        
        # Base prefix for all generated files
        base_prefix = os.path.join(work_dir, "gwas")
        
        # Step 1: Process kinship/relatedness if enabled
        qc_prefix = base_prefix
        if params['use_king']:
            king_cmd = (
                f"plink2 --bed {bed_file} --bim {bim_file} --fam {fam_file} "
                f"--make-king-table --out {base_prefix}_king"
            )
            subprocess.run(king_cmd, shell=True, check=True)
            
            # Apply king-cutoff to remove related individuals
            king_cutoff_cmd = (
                f"plink2 --bed {bed_file} --bim {bim_file} --fam {fam_file} "
                f"--king-cutoff {base_prefix}_king.kin0 {params['king_cutoff']} "
                f"--make-bed --out {base_prefix}_unrelated"
            )
            subprocess.run(king_cutoff_cmd, shell=True, check=True)
            
            # Update input files to use the unrelated subset
            bed_file = f"{base_prefix}_unrelated.bed"
            bim_file = f"{base_prefix}_unrelated.bim"
            fam_file = f"{base_prefix}_unrelated.fam"
            qc_prefix = f"{base_prefix}_unrelated"
        
        # Step 2: Quality control using specified parameters
        qc_cmd = (
            f"plink2 --bed {bed_file} --bim {bim_file} --fam {fam_file} "
            f"--maf {params['maf']} --hwe {params['hwe']} --geno {params['geno']} --mind {params['mind']} "
            f"--make-bed --out {qc_prefix}_qc"
        )
        subprocess.run(qc_cmd, shell=True, check=True)
        
        # Step 3: Calculate PCs if needed for covariates
        if any(covar.startswith('PC') for covar in params['selected_covars']):
            # First handle multicollinearity in SNPs used for PCA
            multicollinearity_cmd = ""
            if params['use_vif']:
                multicollinearity_cmd = f"--vif {params['vif_threshold']}"
            else:
                multicollinearity_cmd = f"--max-corr {params['max_corr']}"
                
            pca_cmd = (
                f"plink2 --bfile {qc_prefix}_qc {multicollinearity_cmd} "
                f"--pca {params['n_pcs']} --out {base_prefix}_pca"
            )
            subprocess.run(pca_cmd, shell=True, check=True)
            
            # Prepare covariate file with selected PCs
            pcs_df = pd.read_csv(f"{base_prefix}_pca.eigenvec", sep='\s+', header=None)
            pcs_df.columns = ['FID', 'IID'] + [f'PC{i+1}' for i in range(pcs_df.shape[1]-2)]
            
            # Merge with original covariates (age, sex, etc.)
            orig_covars_df = pd.read_csv(covar_file, sep='\s+')
            merged_covars = pd.merge(pcs_df[['FID', 'IID'] + [pc for pc in params['selected_covars'] if pc.startswith('PC')]], 
                                   orig_covars_df, on=['FID', 'IID'])
            
            # Save merged covariates
            merged_covars.to_csv(f"{base_prefix}_covars.txt", sep='\t', index=False)
            covar_file = f"{base_prefix}_covars.txt"
        
        # Step 4: Run GWAS with selected parameters
        # Prepare covariate parameter
        covar_param = ""
        if params['selected_covars']:
            covar_param = f"--covar {covar_file} --covar-name " + ",".join(params['selected_covars'])
        
        # Prepare Firth correction parameter
        firth_param = ""
        if params['model_type'] == 'logistic':
            if params['firth_approach'] == 'firth':
                firth_param = "--firth"
            elif params['firth_approach'] == 'firth-fallback':
                firth_param = "--glm firth-fallback"
        
        # Prepare adjust method parameter
        adjust_param = ""
        if params['adjust_method'] != 'none':
            adjust_param = f"--adjust {params['adjust_method']}"
        
        # Prepare dosage parameter
        dosage_param = ""
        if params['use_dosage']:
            if os.path.exists(dosage_file):
                dosage_param = f"--import-dosage {dosage_file} format=oxford ref-first"
                dosage_param += f" dosage-certainty={params['dosage_certainty']}"
        
        # Determine input files based on whether we're using dosage
        input_files = ""
        if params['use_dosage']:
            # For dosage mode
            input_files = dosage_param
            # Additional file to list samples after QC might be needed here
        else:
            # For regular genotype mode
            input_files = f"--bfile {qc_prefix}_qc"
        
        # Prepare model type parameter
        model_param = f"--{params['model_type']}"
        
        # Run GWAS
        gwas_cmd = (
            f"plink2 {input_files} {covar_param} {firth_param} {model_param} "
            f"--pheno {pheno_file} --glm hide-covar {adjust_param} "
            f"--out {base_prefix}_results"
        )
        subprocess.run(gwas_cmd, shell=True, check=True)
        
        # Step 5: Evaluate results
        # Determine the correct output file based on model type
        if params['model_type'] == 'linear':
            results_file = f"{base_prefix}_results.PHENO1.glm.linear"
            if params['adjust_method'] != 'none':
                adjusted_file = f"{base_prefix}_results.PHENO1.glm.linear.adjusted"
        else:
            results_file = f"{base_prefix}_results.PHENO1.glm.logistic"
            if params['adjust_method'] != 'none':
                adjusted_file = f"{base_prefix}_results.PHENO1.glm.logistic.adjusted"
        
        if os.path.exists(results_file):
            results = pd.read_csv(results_file, sep='\s+')
            
            # Calculate genomic inflation factor
            observed_p = results['P'].dropna()
            observed_p = observed_p[observed_p > 0]  # Remove p-values of 0
            
            if len(observed_p) > 0:
                observed_chi2 = np.array([-2 * np.log(p) for p in observed_p])
                expected_chi2 = np.array([chi2.ppf((i+0.5)/len(observed_chi2), 1) 
                                       for i in range(len(observed_chi2))])
                
                # Sort for proper comparison
                observed_chi2.sort()
                expected_chi2.sort()
                
                # Calculate lambda
                lambda_gc = np.median(observed_chi2) / np.median(expected_chi2)
                
                # For GWAS, lambda_gc should ideally be close to 1.0
                # Too high (>1.05) indicates inflation, too low indicates loss of power
                lambda_deviation = abs(lambda_gc - 1.0)
                
                # Calculate number of significant hits
                # Use adjusted p-values if available
                if params['adjust_method'] != 'none' and os.path.exists(adjusted_file):
                    adjusted_results = pd.read_csv(adjusted_file, sep='\s+')
                    # Use the column name that corresponds to the chosen method
                    method_col = f"{params['adjust_method'].upper()}"
                    sig_hits = sum(adjusted_results[method_col] < 0.05)
                else:
                    # Use Bonferroni-corrected threshold for raw p-values
                    bonferroni_threshold = 0.05 / len(results)
                    sig_hits = sum(results['P'] < bonferroni_threshold)
                
                # Create a combined metric:
                # - Penalize deviation from lambda=1.0 (inflation/deflation)
                # - Reward significant hits (discovery power)
                # - Reward more significant hits at stricter thresholds (precision)
                
                # Count hits at genome-wide significance (5e-8)
                genome_wide_hits = sum(results['P'] < 5e-8)
                
                # Optional: Calculate additional quality metrics
                # Example: Effect size consistency
                if 'BETA' in results.columns or 'OR' in results.columns:
                    effect_col = 'BETA' if 'BETA' in results.columns else 'OR'
                    sig_results = results[results['P'] < 0.001]
                    effect_consistency = 0
                    
                    if len(sig_results) > 0:
                        # For positive control studies, check correlation with known effects
                        # This is a placeholder - implement based on your validation strategy
                        effect_consistency = 1.0
                
                # Create a weighted performance metric
                # Lower is better since we're minimizing
                performance_metric = (
                    10 * lambda_deviation  # Heavily penalize inflation/deflation
                    - np.log10(sig_hits + 1)  # Reward significant hits
                    - 2 * np.log10(genome_wide_hits + 1)  # Reward genome-wide hits more
                )
                
                # Optional: add additional metrics based on study goals
                # For example, if you want to prioritize finding specific loci:
                # - Add a bonus for discovering known positives
                # - Add a penalty for missing known associations
                
                return performance_metric
        
        # If results file doesn't exist or other issues
        return float('inf')  # Worst possible score
        
    except Exception as e:
        print(f"Error running PLINK2 with parameters: {e}")
        return float('inf')  # Worst possible score
        
    finally:
        # Clean up (comment out for debugging)
        shutil.rmtree(work_dir)

# Create the study and run optimization
study = optuna.create_study(
    study_name="plink2_gwas_optimization",
    storage="sqlite:///plink2_optimization.db",  # Save results in a database
    direction="minimize",  # We want to minimize our performance metric
    load_if_exists=True  # Resume if the study already exists
)

# Run optimization trials
study.optimize(objective, n_trials=50)

# Print results
print("Best parameters:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")
print(f"Best performance: {study.best_value}")

# Generate optimal command line
optimal_params = study.best_params
# This function would generate the optimal plink2 command line
def generate_optimal_command(params):
    # This would construct the full plink2 command with all parameters
    # based on the optimal parameters found
    cmd = "plink2 --bed input.bed --bim input.bim --fam input.fam "
    
    # Add QC filters
    cmd += f"--maf {params['maf']} --hwe {params['hwe']} "
    cmd += f"--geno {params['geno']} --mind {params['mind']} "
    
    # Add kinship if used
    if params.get('use_king', False):
        cmd += f"--king-cutoff {params['king_cutoff']} "
    
    # Add covariates
    if params.get('selected_covars', []):
        cmd += "--covar covariates.txt --covar-name "
        cmd += ",".join(params['selected_covars']) + " "
    
    # Add model type and Firth if applicable
    cmd += f"--{params['model_type']} "
    if params['model_type'] == 'logistic' and params.get('firth_approach') != 'none':
        if params['firth_approach'] == 'firth':
            cmd += "--firth "
        elif params['firth_approach'] == 'firth-fallback':
            cmd += "--glm firth-fallback "
    
    # Add multiple testing correction
    if params.get('adjust_method', 'none') != 'none':
        cmd += f"--adjust {params['adjust_method']} "
    
    # Add dosage parameters if used
    if params.get('use_dosage', False):
        cmd += f"--import-dosage dosage.gz dosage-certainty={params['dosage_certainty']} "
    
    # Add VIF/max-corr
    if params.get('use_vif', False):
        cmd += f"--vif {params['vif_threshold']} "
    else:
        cmd += f"--max-corr {params['max_corr']} "
    
    cmd += "--out results"
    return cmd

# Generate and print the optimal command
optimal_cmd = generate_optimal_command(optimal_params)
print("Optimal PLINK2 command:")
print(optimal_cmd)

# Save visualization
try:
    import matplotlib.pyplot as plt
    from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
    
    # Create visualization directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Optimization history
    fig = plot_optimization_history(study)
    fig.write_image("visualizations/optimization_history.png")
    
    # Parameter importance
    fig = plot_param_importances(study)
    fig.write_image("visualizations/param_importances.png")
    
    # Parallel coordinate plot
    fig = plot_parallel_coordinate(study)
    fig.write_image("visualizations/parallel_coordinate.png")
    
    print("Visualizations saved to 'visualizations' directory")
except Exception as e:
    print(f"Couldn't create visualizations: {e}")




