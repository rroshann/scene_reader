#!/usr/bin/env python3
"""
Cost Analysis for Approach 3.5
Calculate API costs and savings from optimizations
"""
import pandas as pd
from pathlib import Path
from datetime import datetime


# API Pricing (as of 2024, update if needed)
PRICING = {
    'gpt-3.5-turbo': {
        'input': 0.50 / 1_000_000,  # $0.50 per 1M input tokens
        'output': 1.50 / 1_000_000   # $1.50 per 1M output tokens
    },
    'gpt-4o-mini': {
        'input': 0.15 / 1_000_000,  # $0.15 per 1M input tokens
        'output': 0.60 / 1_000_000   # $0.60 per 1M output tokens
    }
}


def load_results():
    """Load Approach 3.5 results"""
    project_root = Path(__file__).parent.parent.parent
    results_file = project_root / 'results' / 'approach_3_5_optimized' / 'raw' / 'batch_results.csv'
    
    if not results_file.exists():
        return None
    
    df = pd.read_csv(results_file)
    return df[df['success'] == True].copy()


def estimate_tokens_from_word_count(word_count):
    """Estimate tokens from word count (rough approximation: 1 token ≈ 0.75 words)"""
    return int(word_count / 0.75)


def calculate_cost(row):
    """Calculate cost for a single query"""
    llm_model = row['llm_model'].lower()
    
    if llm_model not in PRICING:
        return None, None, None
    
    # Estimate tokens if not available
    if pd.notna(row.get('tokens_used')):
        total_tokens = row['tokens_used']
        # Rough split: 70% input, 30% output (approximation)
        input_tokens = int(total_tokens * 0.7)
        output_tokens = int(total_tokens * 0.3)
    elif pd.notna(row.get('word_count')):
        total_tokens = estimate_tokens_from_word_count(row['word_count'])
        input_tokens = int(total_tokens * 0.7)
        output_tokens = int(total_tokens * 0.3)
    else:
        return None, None, None
    
    pricing = PRICING[llm_model]
    input_cost = input_tokens * pricing['input']
    output_cost = output_tokens * pricing['output']
    total_cost = input_cost + output_cost
    
    return total_cost, input_tokens, output_tokens


def analyze_costs(df):
    """Analyze costs for all queries"""
    costs = []
    input_tokens_list = []
    output_tokens_list = []
    
    for _, row in df.iterrows():
        cost, input_tokens, output_tokens = calculate_cost(row)
        if cost is not None:
            costs.append(cost)
            input_tokens_list.append(input_tokens)
            output_tokens_list.append(output_tokens)
        else:
            costs.append(None)
            input_tokens_list.append(None)
            output_tokens_list.append(None)
    
    df['estimated_cost'] = costs
    df['estimated_input_tokens'] = input_tokens_list
    df['estimated_output_tokens'] = output_tokens_list
    
    return df


def main():
    """Main function to perform cost analysis"""
    print("=" * 80)
    print("COST ANALYSIS FOR APPROACH 3.5")
    print("=" * 80)
    print()
    
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / 'results' / 'approach_3_5_optimized' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_results()
    if df is None or len(df) == 0:
        print("No successful results found. Cannot perform cost analysis.")
        return
    
    # Analyze costs
    df = analyze_costs(df)
    
    # Generate report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("APPROACH 3.5: COST ANALYSIS")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Overall cost statistics
    report_lines.append("OVERALL COST STATISTICS")
    report_lines.append("-" * 80)
    total_cost = df['estimated_cost'].sum()
    report_lines.append(f"Total queries: {len(df)}")
    report_lines.append(f"Total estimated cost: ${total_cost:.6f}")
    report_lines.append(f"Cost per query: ${total_cost / len(df):.6f}")
    report_lines.append(f"Cost per 1000 queries: ${(total_cost / len(df) * 1000):.4f}")
    report_lines.append("")
    
    # Cost by LLM model
    report_lines.append("COST BY LLM MODEL")
    report_lines.append("-" * 80)
    for llm in df['llm_model'].unique():
        llm_df = df[df['llm_model'] == llm]
        llm_cost = llm_df['estimated_cost'].sum()
        llm_count = len(llm_df)
        report_lines.append(f"{llm}:")
        report_lines.append(f"  Queries: {llm_count}")
        report_lines.append(f"  Total cost: ${llm_cost:.6f}")
        report_lines.append(f"  Cost per query: ${llm_cost / llm_count:.6f}")
        report_lines.append(f"  Cost per 1000 queries: ${(llm_cost / llm_count * 1000):.4f}")
        report_lines.append("")
    
    # Cost comparison
    gpt35_df = df[df['llm_model'] == 'gpt-3.5-turbo']
    gpt4mini_df = df[df['llm_model'] == 'gpt-4o-mini']
    
    if len(gpt35_df) > 0 and len(gpt4mini_df) > 0:
        gpt35_cost = gpt35_df['estimated_cost'].sum()
        gpt4mini_cost = gpt4mini_df['estimated_cost'].sum()
        
        report_lines.append("COST COMPARISON")
        report_lines.append("-" * 80)
        report_lines.append(f"GPT-3.5-turbo total cost: ${gpt35_cost:.6f}")
        report_lines.append(f"GPT-4o-mini total cost: ${gpt4mini_cost:.6f}")
        
        if gpt4mini_cost > 0:
            cost_savings = gpt4mini_cost - gpt35_cost
            cost_savings_pct = (cost_savings / gpt4mini_cost) * 100
            report_lines.append(f"Cost savings: ${cost_savings:.6f} ({cost_savings_pct:.1f}%)")
        
        report_lines.append("")
        
        # Per query comparison
        gpt35_per_query = gpt35_cost / len(gpt35_df)
        gpt4mini_per_query = gpt4mini_cost / len(gpt4mini_df)
        report_lines.append(f"GPT-3.5-turbo cost per query: ${gpt35_per_query:.6f}")
        report_lines.append(f"GPT-4o-mini cost per query: ${gpt4mini_per_query:.6f}")
        if gpt4mini_per_query > 0:
            savings_per_query = gpt4mini_per_query - gpt35_per_query
            savings_pct = (savings_per_query / gpt4mini_per_query) * 100
            report_lines.append(f"Savings per query: ${savings_per_query:.6f} ({savings_pct:.1f}%)")
        report_lines.append("")
    
    # Token usage statistics
    if 'estimated_input_tokens' in df.columns:
        report_lines.append("TOKEN USAGE STATISTICS")
        report_lines.append("-" * 80)
        total_input_tokens = df['estimated_input_tokens'].sum()
        total_output_tokens = df['estimated_output_tokens'].sum()
        total_tokens = total_input_tokens + total_output_tokens
        
        report_lines.append(f"Total input tokens: {total_input_tokens:,}")
        report_lines.append(f"Total output tokens: {total_output_tokens:,}")
        report_lines.append(f"Total tokens: {total_tokens:,}")
        report_lines.append(f"Average tokens per query: {total_tokens / len(df):.0f}")
        report_lines.append("")
        
        # By model
        for llm in df['llm_model'].unique():
            llm_df = df[df['llm_model'] == llm]
            llm_input = llm_df['estimated_input_tokens'].sum()
            llm_output = llm_df['estimated_output_tokens'].sum()
            llm_total = llm_input + llm_output
            report_lines.append(f"{llm}:")
            report_lines.append(f"  Input tokens: {llm_input:,}")
            report_lines.append(f"  Output tokens: {llm_output:,}")
            report_lines.append(f"  Total tokens: {llm_total:,}")
            report_lines.append(f"  Avg per query: {llm_total / len(llm_df):.0f}")
            report_lines.append("")
    
    # Cache impact (if cache hits exist)
    if 'cache_hit' in df.columns:
        cache_hits = df[df['cache_hit'] == True]
        if len(cache_hits) > 0:
            report_lines.append("CACHE IMPACT ON COSTS")
            report_lines.append("-" * 80)
            report_lines.append(f"Cache hits: {len(cache_hits)}")
            report_lines.append(f"Cache hit rate: {100 * len(cache_hits) / len(df):.1f}%")
            
            # Estimate cost savings from cache
            # Cache hits avoid LLM generation, saving generation costs
            cache_hit_cost = cache_hits['estimated_cost'].sum()
            cache_miss_cost = df[df['cache_hit'] == False]['estimated_cost'].sum()
            
            report_lines.append(f"Cost with cache hits: ${cache_hit_cost:.6f}")
            report_lines.append(f"Cost without cache (estimated): ${cache_miss_cost + cache_hit_cost:.6f}")
            report_lines.append("")
    
    # Pricing reference
    report_lines.append("PRICING REFERENCE")
    report_lines.append("-" * 80)
    report_lines.append("GPT-3.5-turbo:")
    report_lines.append(f"  Input: ${PRICING['gpt-3.5-turbo']['input'] * 1_000_000:.2f} per 1M tokens")
    report_lines.append(f"  Output: ${PRICING['gpt-3.5-turbo']['output'] * 1_000_000:.2f} per 1M tokens")
    report_lines.append("GPT-4o-mini:")
    report_lines.append(f"  Input: ${PRICING['gpt-4o-mini']['input'] * 1_000_000:.2f} per 1M tokens")
    report_lines.append(f"  Output: ${PRICING['gpt-4o-mini']['output'] * 1_000_000:.2f} per 1M tokens")
    report_lines.append("")
    report_lines.append("Note: Token counts are estimates based on word count if actual tokens not available.")
    report_lines.append("")
    
    # Write report
    with open(output_dir / 'cost_analysis.txt', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print("Cost analysis complete!")
    print(f"Report saved to: {output_dir / 'cost_analysis.txt'}")


if __name__ == '__main__':
    main()

