import argparse, os, sys, json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agentic_selection.evaluation.metrics import bootstrap_ci

def plot_and_save(fig_func, figs_dir, name):
    try:
        fig = plt.figure(figsize=(10, 6))
        fig_func()
        plt.tight_layout()
        plt.savefig(figs_dir / f'{name}.png', dpi=200)
        if name in ['fig01_regret_by_method_ci', 'fig02_paired_diff_forest', 'fig03_regret_heatmap_task_x_method']:
            plt.savefig(figs_dir / f'{name}.pdf', dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f'Skipped {name}: {e}')
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out', default='results/report')
    args = parser.parse_args()

    out_dir = Path(args.out)
    figs_dir = out_dir / 'figs'
    tabs_dir = out_dir / 'tables'
    figs_dir.mkdir(parents=True, exist_ok=True)
    tabs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df[~df['run_id'].str.contains('tune')]
    
    print('Columns:', df.columns.tolist())

    sns.set_palette('colorblind')

    def f01():
        means = df.groupby('condition')['regret'].mean().sort_values()
        sns.barplot(data=df, x='condition', y='regret', order=means.index, errorbar=None)
        for i, cond in enumerate(means.index):
            sub = df[df['condition'] == cond]['regret']
            if len(sub) > 1:
                ci = bootstrap_ci(sub, df[df['condition'] == cond]['pool_seed'])
                plt.errorbar(i, means[cond], yerr=[[means[cond] - ci.lower], [ci.upper - means[cond]]], color='black', capsize=5)
        plt.title('Mean Regret by Condition')
        plt.ylabel('Regret')
    plot_and_save(f01, figs_dir, 'fig01_regret_by_method_ci')

    def f02():
        agent = df[df['condition'] == 'agent_full']
        baselines = [c for c in df['condition'].unique() if c != 'agent_full']
        diffs = []
        for b in baselines:
            base = df[df['condition'] == b]
            m = agent.merge(base, on=['task_key', 'pool_seed'], suffixes=('', '_base'))
            if not m.empty:
                d = m['regret'] - m['regret_base']
                ci = bootstrap_ci(d, m['pool_seed'])
                diffs.append((b, d.mean(), ci.lower, ci.upper))
        if diffs:
            y = np.arange(len(diffs))
            plt.errorbar([d[1] for d in diffs], y, xerr=[[d[1]-d[2] for d in diffs], [d[3]-d[1] for d in diffs]], fmt='o')
            plt.yticks(y, [d[0] for d in diffs])
            plt.axvline(0, color='r', linestyle='--')
            plt.title('Paired Regret Diff (Agent - Baseline)')
            plt.xlabel('Diff Regret')
    plot_and_save(f02, figs_dir, 'fig02_paired_diff_forest')

    def f03():
        pivot = df.pivot_table(index='task_key', columns='condition', values='regret')
        sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
        plt.title('Regret by Task and Condition')
    plot_and_save(f03, figs_dir, 'fig03_regret_heatmap_task_x_method')

    def f04():
        pivot = df.pivot_table(index='pool_seed', columns='condition', values='regret')
        if 'agent_full' in pivot.columns:
            pivot = pivot.sort_values('agent_full')
        sns.heatmap(pivot, cmap='viridis')
        plt.title('Regret by Pool')
    plot_and_save(f04, figs_dir, 'fig04_regret_heatmap_pool_x_method')

    def f05():
        agent = df[df['condition'] == 'agent_full'].copy()
        if 'weights_json' in agent.columns:
            weights = agent['weights_json'].dropna().apply(json.loads).apply(pd.Series)
            w_df = pd.concat([agent['task_key'], weights], axis=1).groupby('task_key').mean()
            sns.heatmap(w_df, annot=True, cmap='viridis', fmt='.2f')
            plt.title('Agent Weights')
    plot_and_save(f05, figs_dir, 'fig05_weights_heatmap_task_x_criterion_agent')

    def f06():
        lookup = df[df['condition'] == 'lookup_table'].copy()
        if 'weights_json' in lookup.columns:
            weights = lookup['weights_json'].dropna().apply(json.loads).apply(pd.Series)
            w_df = pd.concat([lookup['task_key'], weights], axis=1).groupby('task_key').mean()
            sns.heatmap(w_df, annot=True, cmap='viridis', fmt='.2f')
            plt.title('Lookup Weights')
    plot_and_save(f06, figs_dir, 'fig06_weights_heatmap_task_x_criterion_lookup')

    def f07():
        agent = df[df['condition'] == 'agent_full'].copy()
        lookup = df[df['condition'] == 'lookup_table'].copy()
        if 'weights_json' in agent.columns and 'weights_json' in lookup.columns:
            wa = agent['weights_json'].dropna().apply(json.loads).apply(pd.Series)
            wa = pd.concat([agent['task_key'], wa], axis=1).groupby('task_key').mean()
            wl = lookup['weights_json'].dropna().apply(json.loads).apply(pd.Series)
            wl = pd.concat([lookup['task_key'], wl], axis=1).groupby('task_key').mean()
            delta = wa - wl
            sns.heatmap(delta, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
            plt.title('Weight Delta (Agent - Lookup)')
    plot_and_save(f07, figs_dir, 'fig07_weight_delta_heatmap_agent_minus_lookup')

    def f08():
        sns.ecdfplot(data=df, x='regret', hue='condition')
        plt.title('Regret ECDF')
    plot_and_save(f08, figs_dir, 'fig08_regret_ecdf_by_method')

    def f09():
        conds = df['condition'].unique()
        mat = pd.DataFrame(index=conds, columns=conds, dtype=float)
        for c1 in conds:
            for c2 in conds:
                m = df[df['condition']==c1].merge(df[df['condition']==c2], on=['task_key', 'pool_seed'])
                if not m.empty:
                    mat.loc[c1, c2] = (m['regret_x'] < m['regret_y']).mean()
        sns.heatmap(mat, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Win Rate (Row beats Col)')
    plot_and_save(f09, figs_dir, 'fig09_pairwise_winrate_matrix')

    def f10():
        if 'latency_seconds' in df.columns:
            sns.scatterplot(data=df, x='latency_seconds', y='regret', hue='condition')
            plt.xscale('log')
            plt.title('Latency vs Regret')
    plot_and_save(f10, figs_dir, 'fig10_latency_vs_regret_pareto')

    def f11():
        if 'fallback_triggered' in df.columns:
            sns.barplot(data=df, x='task_key', y='fallback_triggered')
            plt.xticks(rotation=90)
            plt.title('Fallback Rate')
    plot_and_save(f11, figs_dir, 'fig11_fallback_rate_by_task')

    def f12():
        agent = df[df['condition'] == 'agent_full'].copy()
        agent = agent.sort_values('pool_seed')
        agent['rolling_regret'] = agent['regret'].rolling(window=5, min_periods=1).mean()
        sns.lineplot(data=agent, x='pool_seed', y='rolling_regret')
        plt.title('Rolling Regret')
    plot_and_save(f12, figs_dir, 'fig12_learning_curve_regret_vs_round')

    def f13():
        has_direct = df['model'].str.contains('DirectWeight', na=False).any()
        has_class = df['model'].str.contains('Classification', na=False).any()
        if has_direct and has_class:
            sub = df[df['model'].str.contains('DirectWeight|Classification', na=False)].copy()
            sub['reasoner'] = sub['model'].apply(lambda x: 'DirectWeight' if 'DirectWeight' in str(x) else 'Classification')
            sns.barplot(data=sub, x='reasoner', y='regret')
            plt.title('Ablation: Direct vs Classification')
        else:
            print('Skipping fig13: missing DirectWeight or Classification model')
    plot_and_save(f13, figs_dir, 'fig13_ablation_direct_vs_classification')

    t1 = df.groupby('condition')[['regret', 'latency_seconds']].mean().reset_index()
    t1.to_csv(tabs_dir / 'table01_summary.csv', index=False)
    
    # Pairwise agent_full - lookup_table
    pivoted = df.pivot_table(index=['task_key', 'pool_seed'], columns='condition', values='regret').dropna()
    if 'agent_full' in pivoted.columns and 'lookup_table' in pivoted.columns:
        pivoted['diff'] = pivoted['agent_full'] - pivoted['lookup_table']
        pivoted[['agent_full', 'lookup_table', 'diff']].to_csv(tabs_dir / 'table02_paired_tests.csv')
    else:
        pd.DataFrame({'error': ['missing agent_full or lookup_table data']}).to_csv(tabs_dir / 'table02_paired_tests.csv')

    t3 = df.groupby(['task_key', 'condition'])['regret'].mean().unstack().reset_index()
    t3.to_csv(tabs_dir / 'table03_per_task.csv', index=False)

    if df['model'].str.contains('DirectWeight|Classification', na=False).any():
        sub = df[df['model'].str.contains('DirectWeight|Classification', na=False)].copy()
        sub['reasoner'] = sub['model'].apply(lambda x: 'DirectWeight' if 'DirectWeight' in str(x) else 'Classification')
        t4 = sub.groupby('reasoner')[['regret', 'fallback_triggered']].mean().reset_index()
        t4.to_csv(tabs_dir / 'table04_ablation.csv', index=False)
    else:
        pd.DataFrame({'error': ['missing DirectWeight or Classification model data']}).to_csv(tabs_dir / 'table04_ablation.csv', index=False)
        
    t5 = df.groupby('condition')[['latency_seconds', 'total_duration']].mean().reset_index()
    t5.to_csv(tabs_dir / 'table05_runtime.csv', index=False)

    with open(out_dir / 'REPORT.md', 'w') as f:
        m = df.get('model', pd.Series(['unknown'])).iloc[0]
        pools = len(df['pool_seed'].unique())
        
        # Real verdict
        verdict = "Unknown"
        if 'agent_full' in pivoted.columns and 'lookup_table' in pivoted.columns:
            mean_diff = pivoted['diff'].mean()
            if mean_diff < 0:
                verdict = f"Yes (Agent wins by {-mean_diff:.4f})"
            else:
                verdict = f"No (Agent loses by {mean_diff:.4f})"
                
        f.write(f'VERDICT: Agent VS best baseline computed: {verdict}\n')
        f.write(f'Config: model={m}, n_pools={pools}\n')

if __name__ == '__main__':
    main()

