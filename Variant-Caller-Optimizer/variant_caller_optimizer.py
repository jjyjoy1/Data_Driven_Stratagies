import optuna
import subprocess
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('variant_caller_optimization')

class VariantCallerOptimizer:
    def __init__(self, 
                 reference_genome,
                 tumor_bam,
                 normal_bam=None,
                 truth_vcf=None,
                 output_dir="./optimization_results",
                 n_trials=50,
                 n_jobs=-1):
        """
        Initialize variant caller optimizer
        
        Parameters:
        -----------
        reference_genome : str, path to reference genome
        tumor_bam : str, path to tumor BAM file
        normal_bam : str or None, path to normal BAM file
        truth_vcf : str or None, path to truth VCF for evaluation
        output_dir : str, directory to store results
        n_trials : int, number of optimization trials
        n_jobs : int, number of parallel jobs
        """
        self.reference_genome = reference_genome
        self.tumor_bam = tumor_bam
        self.normal_bam = normal_bam
        self.truth_vcf = truth_vcf
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.paired_mode = normal_bam is not None
        
        # Create directories for each caller
        for caller in ['mutect2', 'vardict', 'freebayes', 'lofreq']:
            (self.output_dir / caller).mkdir(exist_ok=True)
        
    def optimize_mutect2(self):
        """Optimize MuTect2 parameters"""
        def objective(trial):
            # Parameter space for MuTect2
            params = {
                'tumor_lod': trial.suggest_float('tumor_lod', 3.0, 10.0),
                'normal_lod': trial.suggest_float('normal_lod', 2.0, 10.0) if self.paired_mode else None,
                'min_base_quality': trial.suggest_int('min_base_quality', 5, 30),
                'pcr_indel_model': trial.suggest_categorical('pcr_indel_model', 
                                                         ['NONE', 'HOSTILE', 'AGGRESSIVE', 'CONSERVATIVE']),
                'germline_resource': 'path/to/germline_resource.vcf',  # Fixed parameter
                'panel_of_normals': 'path/to/pon.vcf' if self.paired_mode else None  # Fixed parameter
            }
            
            # Filter out None parameters
            params = {k: v for k, v in params.items() if v is not None}
            
            # Generate unique output name for this trial
            output_vcf = self.output_dir / 'mutect2' / f"trial_{trial.number}.vcf"
            
            # Build the command
            cmd = ['gatk', 'Mutect2',
                  '-R', self.reference_genome,
                  '-I', self.tumor_bam,
                  '-O', str(output_vcf),
                  '--tumor-lod', str(params['tumor_lod']),
                  '--min-base-quality-score', str(params['min_base_quality']),
                  '--pcr-indel-model', params['pcr_indel_model'],
                  '--germline-resource', params['germline_resource']]
            
            if self.paired_mode and self.normal_bam:
                cmd.extend(['-I', self.normal_bam, 
                           '-normal', 'NORMAL',  # Assuming sample name is NORMAL
                           '--normal-lod', str(params['normal_lod'])])
                
                if 'panel_of_normals' in params:
                    cmd.extend(['--panel-of-normals', params['panel_of_normals']])
            
            # Execute command
            logger.info(f"Running MuTect2 trial {trial.number} with params: {params}")
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error running MuTect2: {e}")
                logger.error(f"Stdout: {e.stdout.decode()}")
                logger.error(f"Stderr: {e.stderr.decode()}")
                return float('-inf')  # Return worst score on failure
            
            # Evaluate results if truth set is available
            if self.truth_vcf:
                metrics = self._evaluate_vcf(output_vcf, self.truth_vcf)
                trial.set_user_attr('precision', metrics['precision'])
                trial.set_user_attr('recall', metrics['recall'])
                trial.set_user_attr('f1', metrics['f1'])
                logger.info(f"Trial {trial.number} metrics: {metrics}")
                return metrics['f1']  # Optimize for F1 score
            else:
                # If no truth set, we can't evaluate, so just return a dummy value
                # In real scenarios, you might want to use other metrics
                return 0
        
        # Create and run the study
        study = optuna.create_study(direction='maximize', 
                                   study_name="mutect2_optimization")
        study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        
        # Save results
        self._save_optimization_results('mutect2', study)
        return study.best_params
    
    def optimize_vardict(self):
        """Optimize VarDict parameters"""
        def objective(trial):
            # Parameter space for VarDict
            params = {
                'min_allele_freq': trial.suggest_float('min_allele_freq', 0.005, 0.05),
                'min_mapping_quality': trial.suggest_int('min_mapping_quality', 10, 60),
                'min_base_quality': trial.suggest_int('min_base_quality', 5, 40),
                'min_reads': trial.suggest_int('min_reads', 2, 10),
                'max_mismatches': trial.suggest_int('max_mismatches', 2, 8)
            }
            
            # Generate unique output name for this trial
            output_vcf = self.output_dir / 'vardict' / f"trial_{trial.number}.vcf"
            
            # Set up VarDict command - exact command depends on your VarDict installation
            # This is a simplified example
            bed_file = "regions.bed"  # Regions to analyze
            
            # VarDict typically requires writing to intermediate files and piping
            vardict_cmd = [
                'vardict-java',
                '-G', self.reference_genome,
                '-f', str(params['min_allele_freq']),
                '-N', 'TUMOR',
                '-b', self.tumor_bam,
                '-c', '1', '-S', '2', '-E', '3',
                '-q', str(params['min_mapping_quality']),
                '-Q', str(params['min_base_quality']),
                '-r', str(params['min_reads']),
                '-m', str(params['max_mismatches']),
                bed_file
            ]
            
            if self.paired_mode and self.normal_bam:
                # For tumor-normal mode
                vardict_cmd = [
                    'vardict-java',
                    '-G', self.reference_genome,
                    '-f', str(params['min_allele_freq']),
                    '-N', 'TUMOR',
                    '-b', f"{self.tumor_bam}|{self.normal_bam}",
                    '-c', '1', '-S', '2', '-E', '3',
                    '-q', str(params['min_mapping_quality']),
                    '-Q', str(params['min_base_quality']),
                    bed_file
                ]
            
            # Need to pipe VarDict output through testsomatic or teststrandbias, then var2vcf
            # This is simplified and would need to be adjusted for actual execution
            
            # Execute command (in reality would need to handle the piping properly)
            try:
                # In practice, you'd need a more complex command structure with pipes
                subprocess.run(vardict_cmd, check=True)
                
                # Additional steps to convert output to VCF
                # ...
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Error running VarDict: {e}")
                return float('-inf')
            
            # Evaluate results
            if self.truth_vcf:
                metrics = self._evaluate_vcf(output_vcf, self.truth_vcf)
                return metrics['f1']
            else:
                return 0
        
        # Create and run the study
        study = optuna.create_study(direction='maximize', 
                                   study_name="vardict_optimization")
        study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        
        # Save results
        self._save_optimization_results('vardict', study)
        return study.best_params
    
    def optimize_freebayes(self):
        """Optimize FreeBayes parameters"""
        def objective(trial):
            # Parameter space for FreeBayes
            params = {
                'min_alternate_fraction': trial.suggest_float('min_alternate_fraction', 0.01, 0.2),
                'min_base_quality': trial.suggest_int('min_base_quality', 1, 30),
                'min_mapping_quality': trial.suggest_int('min_mapping_quality', 5, 60),
                'min_alternate_count': trial.suggest_int('min_alternate_count', 2, 10),
                'use_best_n_alleles': trial.suggest_int('use_best_n_alleles', 2, 8)
            }
            
            # Generate unique output name for this trial
            output_vcf = self.output_dir / 'freebayes' / f"trial_{trial.number}.vcf"
            
            # Build command
            cmd = [
                'freebayes',
                '-f', self.reference_genome,
                '--min-alternate-fraction', str(params['min_alternate_fraction']),
                '--min-base-quality', str(params['min_base_quality']),
                '--min-mapping-quality', str(params['min_mapping_quality']),
                '--min-alternate-count', str(params['min_alternate_count']),
                '--use-best-n-alleles', str(params['use_best_n_alleles']),
                '-v', str(output_vcf)
            ]
            
            # Add BAM files
            cmd.append(self.tumor_bam)
            if self.paired_mode and self.normal_bam:
                cmd.append(self.normal_bam)
            
            # Execute command
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error running FreeBayes: {e}")
                return float('-inf')
            
            # Evaluate results
            if self.truth_vcf:
                metrics = self._evaluate_vcf(output_vcf, self.truth_vcf)
                return metrics['f1']
            else:
                return 0
        
        # Create and run the study
        study = optuna.create_study(direction='maximize', 
                                   study_name="freebayes_optimization")
        study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        
        # Save results
        self._save_optimization_results('freebayes', study)
        return study.best_params
    
    def optimize_lofreq(self):
        """Optimize LoFreq parameters"""
        def objective(trial):
            # Parameter space for LoFreq
            params = {
                'min_coverage': trial.suggest_int('min_coverage', 5, 50),
                'min_base_quality': trial.suggest_int('min_base_quality', 10, 30),
                'significance': trial.suggest_float('significance', 0.001, 0.05, log=True),
                'min_alt_count': trial.suggest_int('min_alt_count', 2, 10),
                'max_depth': trial.suggest_int('max_depth', 1000, 10000)
            }
            
            # Generate unique output name for this trial
            output_vcf = self.output_dir / 'lofreq' / f"trial_{trial.number}.vcf"
            
            # Build command for tumor-only mode
            cmd = [
                'lofreq', 'call-parallel',
                '--pp-threads', '4',
                '-f', self.reference_genome,
                '-o', str(output_vcf),
                '-q', str(params['min_base_quality']),
                '-Q', str(params['min_base_quality']),
                '-C', str(params['min_coverage']),
                '-a', str(params['min_alt_count']),
                '-B', str(params['significance']),
                '-d', str(params['max_depth']),
                self.tumor_bam
            ]
            
            if self.paired_mode and self.normal_bam:
                # For paired analysis, we'd need to run lofreq somatic
                # This is a simplified version of the command
                cmd = [
                    'lofreq', 'somatic',
                    '--tumor', self.tumor_bam,
                    '--normal', self.normal_bam,
                    '-o', str(self.output_dir / 'lofreq'),
                    '-f', self.reference_genome,
                    '--min-cov', str(params['min_coverage']),
                    '--min-bq', str(params['min_base_quality']),
                    '--sig', str(params['significance']),
                    '--call-indels'
                ]
                
                # The output would be different for somatic mode
                output_vcf = self.output_dir / 'lofreq' / 'somatic_final.vcf'
            
            # Execute command
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error running LoFreq: {e}")
                return float('-inf')
            
            # Evaluate results
            if self.truth_vcf:
                metrics = self._evaluate_vcf(output_vcf, self.truth_vcf)
                return metrics['f1']
            else:
                return 0
        
        # Create and run the study
        study = optuna.create_study(direction='maximize', 
                                   study_name="lofreq_optimization")
        study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        
        # Save results
        self._save_optimization_results('lofreq', study)
        return study.best_params
    
    def optimize_all(self):
        """Optimize all variant callers"""
        results = {}
        results['mutect2'] = self.optimize_mutect2()
        results['vardict'] = self.optimize_vardict()
        results['freebayes'] = self.optimize_freebayes()
        results['lofreq'] = self.optimize_lofreq()
        
        # Save all results to a summary file
        with open(self.output_dir / 'optimization_summary.txt', 'w') as f:
            for caller, params in results.items():
                f.write(f"Best parameters for {caller}:\n")
                for param, value in params.items():
                    f.write(f"  {param}: {value}\n")
                f.write("\n")
        
        return results
    
    def _evaluate_vcf(self, test_vcf, truth_vcf):
        """
        Evaluate variants against truth set
        
        In practice, you'd use tools like hap.py or custom scripts
        that account for variant normalization, overlapping variants, etc.
        """
        # This is a placeholder for actual evaluation code
        # In a real implementation, you would use a tool like hap.py
        
        # Example command for hap.py:
        cmd = [
            'hap.py',
            truth_vcf,
            test_vcf,
            '-r', self.reference_genome,
            '-o', f"{test_vcf.stem}_eval",
            '--engine', 'vcfeval'
        ]
        
        try:
            subprocess.run(cmd, check=True)
            # Parse the results from hap.py output files
            # and return precision, recall, F1
            
            # Placeholder values
            metrics = {
                'precision': 0.9,
                'recall': 0.8,
                'f1': 0.85
            }
            return metrics
        except:
            # Return poor performance if evaluation fails
            return {'precision': 0, 'recall': 0, 'f1': 0}
    
    def _save_optimization_results(self, caller, study):
        """Save optimization results to files"""
        # Save best parameters
        with open(self.output_dir / caller / 'best_params.txt', 'w') as f:
            f.write(f"Best {caller} parameters:\n")
            for param, value in study.best_params.items():
                f.write(f"{param}: {value}\n")
            f.write(f"Best F1 score: {study.best_value}\n")
        
        # Save all trials information
        trials_df = study.trials_dataframe()
        trials_df.to_csv(self.output_dir / caller / 'all_trials.csv', index=False)
        
        # Create visualization plots
        try:
            optuna.visualization.plot_optimization_history(study).write_html(
                str(self.output_dir / caller / 'optimization_history.html'))
            
            optuna.visualization.plot_param_importances(study).write_html(
                str(self.output_dir / caller / 'param_importances.html'))
            
            optuna.visualization.plot_contour(study).write_html(
                str(self.output_dir / caller / 'contour.html'))
        except:
            logger.warning(f"Failed to create some visualization plots for {caller}")


# Example usage
if __name__ == "__main__":
    # Path configuration - replace with your actual paths
    reference_genome = "/path/to/reference.fa"
    tumor_bam = "/path/to/tumor.bam"
    normal_bam = "/path/to/normal.bam"  # Set to None for tumor-only mode
    truth_vcf = "/path/to/truth.vcf"  # Optional, for evaluation
    
    # Initialize optimizer for tumor-normal paired analysis
    optimizer = VariantCallerOptimizer(
        reference_genome=reference_genome,
        tumor_bam=tumor_bam,
        normal_bam=normal_bam,  # Remove this or set to None for tumor-only
        truth_vcf=truth_vcf,
        output_dir="./variant_caller_optimization",
        n_trials=50  # Adjust based on computational resources
    )
    
    # Optimize a specific caller
    best_mutect2_params = optimizer.optimize_mutect2()
    print(f"Best MuTect2 parameters: {best_mutect2_params}")
    
    # Or optimize all callers
    all_best_params = optimizer.optimize_all()
    print("Optimization complete for all variant callers!")




