#!/usr/bin/env python3
"""
Statistical significance testing for Approach 3.5
Comprehensive statistical analysis of optimizations
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from datetime import datetime


def load_results():
    """Load Approach 3.5 results"""
    project_root = Path(__file__).parent.parent.parent
    results_file = project_root / 'results' / 'approach_3_5_optimized' / 'raw' / 'batch_results.csv'
    
    if not results_file.exists():
        return None
    
    df = pd.read_csv(results_file)
    return df[df['success'] == True].copy()


def load_approach_3_results():
    """Load Approach 3 results for comparison"""
    project_root = Path(__file__).parent.parent.parent
    results_file = project_root / 'results' / 'approach_3_specialized' / 'raw' / 'batch_results.csv'
    
    if not results_file.exists():
        return None
    
    df = pd.read_csv(results_file)
    return df[df['success'] == True].copy()


def paired_t_test(df_3_5, df_3):
    """Paired t-test: Approach 3.5 vs Approach 3 (same images)"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PAIRED T-TEST: APPROACH 3.5 vs APPROACH 3")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    if df_3 is None or len(df_3) == 0:
        report_lines.append("Approach 3 results not available for comparison.")
        report_lines.append("")
        return report_lines
    
    # Filter depth mode only (for fair comparison)
    depth_3_5 = df_3_5[df_3_5['mode'] == 'depth']
    depth_3 = df_3[df_3['mode'] == 'depth']
    
    if len(depth_3_5) == 0 or len(depth_3) == 0:
        report_lines.append("Insufficient data for paired t-test.")
        report_lines.append("")
        return report_lines
    
    # Get common filenames
    common_files = set(depth_3_5['filename']) & set(depth_3['filename'])
    
    if len(common_files) == 0:
        report_lines.append("No common images found for paired comparison.")
        report_lines.append("")
        return report_lines
    
    # Get paired latencies
    latencies_3_5 = []
    latencies_3 = []
    
    for filename in common_files:
        lat_3_5 = depth_3_5[depth_3_5['filename'] == filename]['total_latency'].values
        lat_3 = depth_3[depth_3['filename'] == filename]['total_latency'].values
        
        if len(lat_3_5) > 0 and len(lat_3) > 0:
            latencies_3_5.append(lat_3_5[0])
            latencies_3.append(lat_3[0])
    
    if len(latencies_3_5) < 2:
        report_lines.append("Insufficient paired data for t-test.")
        report_lines.append("")
        return report_lines
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(latencies_3_5, latencies_3)
    
    # Calculate effect size (Cohen's d)
    differences = np.array(latencies_3_5) - np.array(latencies_3)
    cohens_d = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences, ddof=1) > 0 else 0
    
    report_lines.append(f"Common images: {len(latencies_3_5)}")
    report_lines.append(f"Approach 3.5 mean: {np.mean(latencies_3_5):.3f}s")
    report_lines.append(f"Approach 3 mean: {np.mean(latencies_3):.3f}s")
    report_lines.append(f"Mean difference: {np.mean(differences):.3f}s")
    report_lines.append(f"Improvement: {abs(np.mean(differences) / np.mean(latencies_3) * 100):.1f}%")
    report_lines.append("")
    report_lines.append(f"t-statistic: {t_stat:.4f}")
    report_lines.append(f"p-value: {p_value:.6f}")
    report_lines.append(f"Cohen's d (effect size): {cohens_d:.4f}")
    report_lines.append("")
    
    if p_value < 0.001:
        report_lines.append("Result: HIGHLY SIGNIFICANT (p < 0.001)")
    elif p_value < 0.01:
        report_lines.append("Result: VERY SIGNIFICANT (p < 0.01)")
    elif p_value < 0.05:
        report_lines.append("Result: SIGNIFICANT (p < 0.05)")
    else:
        report_lines.append("Result: NOT SIGNIFICANT (p >= 0.05)")
    
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    
    report_lines.append(f"Effect size interpretation: {effect_size}")
    report_lines.append("")
    
    return report_lines


def configuration_anova(df_3_5):
    """One-way ANOVA: Configuration differences"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ONE-WAY ANOVA: CONFIGURATION DIFFERENCES")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    depth_3_5 = df_3_5[df_3_5['mode'] == 'depth']
    
    if len(depth_3_5) == 0:
        report_lines.append("No depth mode data available.")
        report_lines.append("")
        return report_lines
    
    # Group by configuration
    configs = depth_3_5['configuration'].unique()
    
    if len(configs) < 2:
        report_lines.append("Insufficient configurations for ANOVA.")
        report_lines.append("")
        return report_lines
    
    groups = [depth_3_5[depth_3_5['configuration'] == config]['total_latency'].dropna().values 
              for config in configs]
    
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    report_lines.append(f"Configurations: {len(configs)}")
    for i, config in enumerate(configs):
        group_data = groups[i]
        report_lines.append(f"  {config}:")
        report_lines.append(f"    n = {len(group_data)}")
        report_lines.append(f"    Mean = {np.mean(group_data):.3f}s")
        report_lines.append(f"    Std = {np.std(group_data):.3f}s")
    report_lines.append("")
    report_lines.append(f"F-statistic: {f_stat:.4f}")
    report_lines.append(f"p-value: {p_value:.6f}")
    report_lines.append("")
    
    if p_value < 0.001:
        report_lines.append("Result: HIGHLY SIGNIFICANT (p < 0.001)")
    elif p_value < 0.01:
        report_lines.append("Result: VERY SIGNIFICANT (p < 0.01)")
    elif p_value < 0.05:
        report_lines.append("Result: SIGNIFICANT (p < 0.05)")
    else:
        report_lines.append("Result: NOT SIGNIFICANT (p >= 0.05)")
    report_lines.append("")
    
    return report_lines


def llm_model_t_test(df_3_5):
    """T-test: GPT-3.5-turbo vs GPT-4o-mini"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("T-TEST: GPT-3.5-TURBO vs GPT-4O-MINI")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    depth_3_5 = df_3_5[df_3_5['mode'] == 'depth']
    
    gpt35 = depth_3_5[depth_3_5['llm_model'] == 'gpt-3.5-turbo']['total_latency'].dropna()
    gpt4mini = depth_3_5[depth_3_5['llm_model'] == 'gpt-4o-mini']['total_latency'].dropna()
    
    if len(gpt35) < 2 or len(gpt4mini) < 2:
        report_lines.append("Insufficient data for t-test.")
        report_lines.append("")
        return report_lines
    
    # Perform independent t-test
    t_stat, p_value = stats.ttest_ind(gpt35, gpt4mini)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(gpt35) - 1) * gpt35.std()**2 + (len(gpt4mini) - 1) * gpt4mini.std()**2) / 
                         (len(gpt35) + len(gpt4mini) - 2))
    cohens_d = (gpt35.mean() - gpt4mini.mean()) / pooled_std if pooled_std > 0 else 0
    
    report_lines.append(f"GPT-3.5-turbo:")
    report_lines.append(f"  n = {len(gpt35)}")
    report_lines.append(f"  Mean = {gpt35.mean():.3f}s")
    report_lines.append(f"  Std = {gpt35.std():.3f}s")
    report_lines.append("")
    report_lines.append(f"GPT-4o-mini:")
    report_lines.append(f"  n = {len(gpt4mini)}")
    report_lines.append(f"  Mean = {gpt4mini.mean():.3f}s")
    report_lines.append(f"  Std = {gpt4mini.std():.3f}s")
    report_lines.append("")
    report_lines.append(f"Mean difference: {gpt35.mean() - gpt4mini.mean():.3f}s")
    report_lines.append(f"Improvement: {abs((gpt35.mean() - gpt4mini.mean()) / gpt4mini.mean() * 100):.1f}%")
    report_lines.append("")
    report_lines.append(f"t-statistic: {t_stat:.4f}")
    report_lines.append(f"p-value: {p_value:.6f}")
    report_lines.append(f"Cohen's d (effect size): {cohens_d:.4f}")
    report_lines.append("")
    
    if p_value < 0.001:
        report_lines.append("Result: HIGHLY SIGNIFICANT (p < 0.001)")
    elif p_value < 0.01:
        report_lines.append("Result: VERY SIGNIFICANT (p < 0.01)")
    elif p_value < 0.05:
        report_lines.append("Result: SIGNIFICANT (p < 0.05)")
    else:
        report_lines.append("Result: NOT SIGNIFICANT (p >= 0.05)")
    
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    
    report_lines.append(f"Effect size interpretation: {effect_size}")
    report_lines.append("")
    
    return report_lines


def complexity_anova(df_3_5):
    """ANOVA: Complexity-based latency differences"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ANOVA: COMPLEXITY-BASED LATENCY DIFFERENCES")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    depth_3_5 = df_3_5[df_3_5['mode'] == 'depth']
    
    if 'complexity' not in depth_3_5.columns or depth_3_5['complexity'].isna().all():
        report_lines.append("Complexity data not available.")
        report_lines.append("")
        return report_lines
    
    complexities = depth_3_5['complexity'].dropna().unique()
    
    if len(complexities) < 2:
        report_lines.append("Insufficient complexity levels for ANOVA.")
        report_lines.append("")
        return report_lines
    
    groups = [depth_3_5[depth_3_5['complexity'] == comp]['total_latency'].dropna().values 
              for comp in complexities]
    
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    report_lines.append(f"Complexity levels: {len(complexities)}")
    for i, comp in enumerate(complexities):
        group_data = groups[i]
        report_lines.append(f"  {comp}:")
        report_lines.append(f"    n = {len(group_data)}")
        report_lines.append(f"    Mean = {np.mean(group_data):.3f}s")
        report_lines.append(f"    Std = {np.std(group_data):.3f}s")
    report_lines.append("")
    report_lines.append(f"F-statistic: {f_stat:.4f}")
    report_lines.append(f"p-value: {p_value:.6f}")
    report_lines.append("")
    
    if p_value < 0.001:
        report_lines.append("Result: HIGHLY SIGNIFICANT (p < 0.001)")
    elif p_value < 0.01:
        report_lines.append("Result: VERY SIGNIFICANT (p < 0.01)")
    elif p_value < 0.05:
        report_lines.append("Result: SIGNIFICANT (p < 0.05)")
    else:
        report_lines.append("Result: NOT SIGNIFICANT (p >= 0.05)")
    report_lines.append("")
    
    return report_lines


def generation_latency_test(df_3_5):
    """T-test: Generation latency improvement"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("T-TEST: GENERATION LATENCY IMPROVEMENT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    depth_3_5 = df_3_5[df_3_5['mode'] == 'depth']
    
    gpt35_gen = depth_3_5[depth_3_5['llm_model'] == 'gpt-3.5-turbo']['generation_latency'].dropna()
    gpt4mini_gen = depth_3_5[depth_3_5['llm_model'] == 'gpt-4o-mini']['generation_latency'].dropna()
    
    if len(gpt35_gen) < 2 or len(gpt4mini_gen) < 2:
        report_lines.append("Insufficient data for t-test.")
        report_lines.append("")
        return report_lines
    
    # Perform independent t-test
    t_stat, p_value = stats.ttest_ind(gpt35_gen, gpt4mini_gen)
    
    # Calculate effect size
    pooled_std = np.sqrt(((len(gpt35_gen) - 1) * gpt35_gen.std()**2 + (len(gpt4mini_gen) - 1) * gpt4mini_gen.std()**2) / 
                         (len(gpt35_gen) + len(gpt4mini_gen) - 2))
    cohens_d = (gpt35_gen.mean() - gpt4mini_gen.mean()) / pooled_std if pooled_std > 0 else 0
    
    report_lines.append(f"GPT-3.5-turbo generation:")
    report_lines.append(f"  Mean = {gpt35_gen.mean():.3f}s")
    report_lines.append(f"  Std = {gpt35_gen.std():.3f}s")
    report_lines.append("")
    report_lines.append(f"GPT-4o-mini generation:")
    report_lines.append(f"  Mean = {gpt4mini_gen.mean():.3f}s")
    report_lines.append(f"  Std = {gpt4mini_gen.std():.3f}s")
    report_lines.append("")
    report_lines.append(f"Mean difference: {gpt35_gen.mean() - gpt4mini_gen.mean():.3f}s")
    report_lines.append(f"Improvement: {abs((gpt35_gen.mean() - gpt4mini_gen.mean()) / gpt4mini_gen.mean() * 100):.1f}%")
    report_lines.append("")
    report_lines.append(f"t-statistic: {t_stat:.4f}")
    report_lines.append(f"p-value: {p_value:.6f}")
    report_lines.append(f"Cohen's d: {cohens_d:.4f}")
    report_lines.append("")
    
    if p_value < 0.001:
        report_lines.append("Result: HIGHLY SIGNIFICANT (p < 0.001)")
    elif p_value < 0.01:
        report_lines.append("Result: VERY SIGNIFICANT (p < 0.01)")
    elif p_value < 0.05:
        report_lines.append("Result: SIGNIFICANT (p < 0.05)")
    else:
        report_lines.append("Result: NOT SIGNIFICANT (p >= 0.05)")
    report_lines.append("")
    
    return report_lines


def main():
    """Main function to run all statistical tests"""
    print("=" * 80)
    print("STATISTICAL TESTS FOR APPROACH 3.5")
    print("=" * 80)
    print()
    
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / 'results' / 'approach_3_5_optimized' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df_3_5 = load_results()
    if df_3_5 is None or len(df_3_5) == 0:
        print("No successful results found. Cannot perform statistical tests.")
        return
    
    df_3 = load_approach_3_results()
    
    # Run all tests
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("APPROACH 3.5: STATISTICAL SIGNIFICANCE TESTS")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Paired t-test
    report_lines.extend(paired_t_test(df_3_5, df_3))
    
    # Configuration ANOVA
    report_lines.extend(configuration_anova(df_3_5))
    
    # LLM model t-test
    report_lines.extend(llm_model_t_test(df_3_5))
    
    # Complexity ANOVA
    report_lines.extend(complexity_anova(df_3_5))
    
    # Generation latency test
    report_lines.extend(generation_latency_test(df_3_5))
    
    # Summary
    report_lines.append("=" * 80)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("All statistical tests completed.")
    report_lines.append("See individual test results above for detailed analysis.")
    report_lines.append("")
    
    # Write report
    with open(output_dir / 'statistical_tests.txt', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print("Statistical tests complete!")
    print(f"Report saved to: {output_dir / 'statistical_tests.txt'}")


if __name__ == '__main__':
    main()

