"""
实验结果可视化脚本

自动扫描 results/ 目录下的实验结果，绘制训练曲线，
并生成预测对比 HTML 表格。

Results 目录结构：
    results/
    ├── bilstm_lr_0.001_bs_32/
    │   ├── training_metrics.csv   (epoch,train_loss,train_acc,val_loss,val_acc)
    │   └── prediction_comparison.csv  (text,true_label,pred_label,correct)
    ├── bilstm_lr_0.0005_bs_32/
    │   └── ...
    ├── finbert_lr_2e-5_bs_16/
    │   └── ...
    └── ...

用法：
    python scripts/plot_results.py
    python scripts/plot_results.py --results_dir results --output results/plots
"""

import os
import re
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import warnings
warnings.filterwarnings('ignore')

# ==================== 颜色与样式配置 ====================

MODEL_COLORS = {
    'bilstm':     '#1f77b4',
    'BiLSTM':     '#1f77b4',
    'finbert':    '#ff7f0e',
    'FinBERT':    '#ff7f0e',
}

LR_BS_LINESTYLE = {
    # (lr, bs) -> (linestyle, marker)
    ('1e-3',   '32'):  ('-',  'o'),
    ('5e-4',   '32'):  ('--', 's'),
    ('2e-4',   '32'):  (':',  '^'),
    ('1e-3',   '16'):  ('-.', 'D'),
    ('2e-5',   '16'):  ('-',  'v'),
    ('2e-5',   '8'):   ('--', 'P'),
}


def _get_style(lr, bs):
    """获取 lr/bs 对应的线型和标记样式"""
    lr_key = str(lr).lower()
    bs_key = str(bs)
    for (lk, bk), style in LR_BS_LINESTYLE.items():
        if lk in lr_key and bk == bs_key:
            return style
    return ('-', 'o')


# ==================== 目录扫描 ====================

def discover_experiments(results_dir):
    """
    扫描 results_dir，收集所有实验的结果目录。

    Returns:
        List[dict]  每个 dict 包含:
            path, model, lr, bs, metrics_df, pred_df, label_map
    """
    experiments = []

    # 扫描所有子目录（含 training_metrics.csv 的目录）
    pattern = os.path.join(results_dir, '*', 'training_metrics.csv')
    csv_files = glob.glob(pattern)

    if not csv_files:
        # 回退：根目录有 metrics 文件（单实验场景）
        root_metrics = os.path.join(results_dir, 'training_metrics.csv')
        if os.path.exists(root_metrics):
            csv_files = [root_metrics]

    for csv_path in csv_files:
        exp_dir = os.path.dirname(csv_path)
        exp_name = os.path.basename(exp_dir)

        # 解析目录名，提取 model / lr / bs
        model, lr, bs = _parse_exp_name(exp_name)
        if model is None:
            model = 'exp'

        try:
            metrics_df = pd.read_csv(csv_path)
        except Exception:
            continue

        pred_csv = os.path.join(exp_dir, 'prediction_comparison.csv')
        pred_df = pd.read_csv(pred_csv) if os.path.exists(pred_csv) else None

        experiments.append({
            'name':     exp_name,
            'model':    model,
            'lr':       lr,
            'bs':       bs,
            'path':     exp_dir,
            'metrics':  metrics_df,
            'predictions': pred_df,
        })

    # 按 model -> lr -> bs 排序
    experiments.sort(key=lambda e: (e['model'], float(e['lr']) if e['lr'] else 0, int(e['bs']) if e['bs'] else 0))
    return experiments


def _parse_exp_name(name):
    """
    从目录名解析 model / lr / bs。
    支持格式如:
        bilstm_lr_0.001_bs_32
        finbert_lr_2e-5_bs_16
        experiment_1
    """
    # 匹配 model_lr_X_bs_Y
    m = re.match(r'([a-zA-Z_]+)_lr_([0-9e.\-]+)_bs_(\d+)', name, re.I)
    if m:
        return m.group(1).lower(), m.group(2), m.group(3)
    return None, None, None


# ==================== 绘图函数 ====================

def set_style():
    """统一图表样式"""
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def plot_all_curves(experiments, output_dir, title_prefix=''):
    """
    绘制 6 张子图，输出到同一 PDF/PNG。

    布局：
        Row 1: BiLSTM Loss  |  FinBERT Loss
        Row 2: BiLSTM Acc   |  FinBERT Acc
        Row 3: All Models Loss (对比) | All Models Acc (对比)
    """
    set_style()

    # 分组
    bilstm_exps = [e for e in experiments if 'lstm' in e['model'].lower()]
    finbert_exps = [e for e in experiments if 'bert' in e['model'].lower()]
    all_exps = experiments

    # 如果没有明确区分，按已有实验分组
    if not bilstm_exps:
        bilstm_exps = all_exps
    if not finbert_exps:
        finbert_exps = all_exps

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{title_prefix}Training & Validation Curves'.strip(), fontsize=15, fontweight='bold', y=1.01)

    # ---------- 第1行: Loss ----------
    _plot_metric(
        axes[0, 0], bilstm_exps,
        train_col='train_loss', val_col='val_loss',
        ylabel='Loss', title='BiLSTM — Training & Validation Loss',
        tag='bilstm',
    )
    _plot_metric(
        axes[0, 1], finbert_exps,
        train_col='train_loss', val_col='val_loss',
        ylabel='Loss', title='FinBERT — Training & Validation Loss',
        tag='finbert',
    )
    _plot_metric(
        axes[0, 2], all_exps,
        train_col='train_loss', val_col='val_loss',
        ylabel='Loss', title='All Models — Validation Loss Comparison',
        tag='all',
    )

    # ---------- 第2行: Accuracy ----------
    _plot_metric(
        axes[1, 0], bilstm_exps,
        train_col='train_acc', val_col='val_acc',
        ylabel='Accuracy', title='BiLSTM — Training & Validation Accuracy',
        tag='bilstm',
    )
    _plot_metric(
        axes[1, 1], finbert_exps,
        train_col='train_acc', val_col='val_acc',
        ylabel='Accuracy', title='FinBERT — Training & Validation Accuracy',
        tag='finbert',
    )
    _plot_metric(
        axes[1, 2], all_exps,
        train_col='train_acc', val_col='val_acc',
        ylabel='Accuracy', title='All Models — Validation Accuracy Comparison',
        tag='all',
    )

    plt.tight_layout()
    out_png = os.path.join(output_dir, 'training_curves.png')
    out_pdf = os.path.join(output_dir, 'training_curves.pdf')
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f'  [Saved] {out_png}')
    print(f'  [Saved] {out_pdf}')
    return out_png


def _plot_metric(ax, experiments, train_col, val_col, ylabel, title, tag):
    """
    在单个 ax 上绘制训练/验证曲线。
    tag == 'all' 时不画训练曲线，只画 val；否则画 train + val。
    """
    if not experiments:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        return

    for exp in experiments:
        df = exp['metrics']
        if train_col not in df.columns or val_col not in df.columns:
            continue

        epochs = df['epoch']
        train_vals = df[train_col]
        val_vals   = df[val_col]

        lr_str = exp['lr']
        bs_str = exp['bs']
        label = f"lr={lr_str}, bs={bs_str}"
        ls, mk = _get_style(lr_str, bs_str)

        if tag == 'all':
            # 只画 val 线
            ax.plot(epochs, val_vals, linestyle=ls, marker=mk, markersize=4,
                    label=label, linewidth=1.8)
        else:
            # 画 val（实线）+ train（虚线）
            ax.plot(epochs, val_vals,   linestyle='-',  marker=mk, markersize=4,
                    label=f"{label} (val)", linewidth=2, color=MODEL_COLORS.get(exp['model'], '#333'))
            if train_col in df.columns:
                ax.plot(epochs, train_vals, linestyle='--', marker=mk, markersize=3,
                        label=f"{label} (train)", linewidth=1.5, alpha=0.6,
                        color=MODEL_COLORS.get(exp['model'], '#333'))

    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', framealpha=0.9)
    ax.set_xlim(left=1)


def plot_separate_figures(experiments, output_dir):
    """
    绘制 6 张独立图片（每张一个子图），方便单独使用。
    文件名：training_loss_bilstm.png 等。
    """
    set_style()

    bilstm_exps = [e for e in experiments if 'lstm' in e['model'].lower()]
    finbert_exps = [e for e in experiments if 'bert' in e['model'].lower()]

    configs = [
        ('training_loss_bilstm.png',     bilstm_exps,  'train_loss', 'val_loss',  'Loss',     'BiLSTM — Training & Validation Loss'),
        ('training_loss_finbert.png',    finbert_exps, 'train_loss', 'val_loss',  'Loss',     'FinBERT — Training & Validation Loss'),
        ('training_loss_comparison.png', experiments,  'train_loss', 'val_loss',  'Loss',     'All Models — Validation Loss Comparison'),
        ('training_acc_bilstm.png',      bilstm_exps,  'train_acc',  'val_acc',   'Accuracy', 'BiLSTM — Training & Validation Accuracy'),
        ('training_acc_finbert.png',     finbert_exps, 'train_acc',  'val_acc',   'Accuracy', 'FinBERT — Training & Validation Accuracy'),
        ('training_acc_comparison.png',  experiments,  'train_acc',  'val_acc',   'Accuracy', 'All Models — Validation Accuracy Comparison'),
    ]

    for filename, exps, tc, vc, ylabel, title in configs:
        fig, ax = plt.subplots(figsize=(10, 6))
        is_all = 'comparison' in filename
        _plot_metric(ax, exps, tc, vc, ylabel, title, 'all' if is_all else 'split')
        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f'  [Saved] {out_path}')


# ==================== HTML 预测表格 ====================

LABEL_COLORS = {
    'neutral':  '#6c757d',
    'positive': '#28a745',
    'negative': '#dc3545',
    '0':        '#6c757d',
    '1':        '#28a745',
    '2':        '#dc3545',
}
LABEL_TEXT_COLOR = {
    'neutral':  '#ffffff',
    'positive': '#ffffff',
    'negative': '#ffffff',
    '0':        '#ffffff',
    '1':        '#ffffff',
    '2':        '#ffffff',
}


def generate_html_table(experiments, output_dir, n_rows=100):
    """
    为每个实验生成一个精美的 HTML 预测对比表格。
    """
    os.makedirs(output_dir, exist_ok=True)

    for exp in experiments:
        pred_df = exp['predictions']
        if pred_df is None or pred_df.empty:
            continue

        n_rows = min(n_rows, len(pred_df))
        df = pred_df.head(n_rows).reset_index(drop=True)

        # 确定标签列名（可能是字符串或整数）
        true_col  = 'true_label'
        pred_col = 'pred_label'

        rows_html = []
        for i, row in df.iterrows():
            idx = i + 1
            text = str(row.get('text', ''))
            true_l = str(row.get(true_col, ''))
            pred_l = str(row.get(pred_col, ''))
            correct = str(row.get('correct', '')).lower() in ('true', '1', 'yes')

            # 截断过长文本
            if len(text) > 120:
                text_display = text[:117] + '...'
            else:
                text_display = text

            # 颜色
            true_bg  = LABEL_COLORS.get(true_l,  '#343a40')
            pred_bg  = LABEL_COLORS.get(pred_l,  '#343a40') if not correct else '#28a745'
            row_bg   = '#f8f9fa' if i % 2 == 0 else '#ffffff'
            status_icon = '&#10004;' if correct else '&#10008;'
            status_color = '#28a745' if correct else '#dc3545'

            rows_html.append(f"""
            <tr style="background:{row_bg};">
              <td class="idx">{idx}</td>
              <td class="text" title="{_esc(text)}">{text_display}</td>
              <td class="label" style="background:{true_bg}; color:white;">{true_l}</td>
              <td class="label" style="background:{pred_bg}; color:white;">{pred_l}</td>
              <td class="status" style="color:{status_color};">{status_icon}</td>
            </tr>
            """)

        rows_str = '\n'.join(rows_html)

        acc = (df[true_col] == df[pred_col]).mean() * 100

        html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Prediction Comparison — {exp['name']}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f0f2f5; padding: 24px; }}
  .container {{ max-width: 1100px; margin: 0 auto; }}

  .header {{
    background: linear-gradient(135deg, #1f77b4, #ff7f0e);
    color: white; border-radius: 12px; padding: 24px 32px;
    margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center;
  }}
  .header h1 {{ font-size: 1.4rem; font-weight: 600; }}
  .header .meta {{
    background: rgba(255,255,255,0.2); border-radius: 8px; padding: 8px 16px;
    text-align: right; font-size: 0.9rem; line-height: 1.6;
  }}

  .stats {{
    display: flex; gap: 16px; margin-bottom: 20px;
  }}
  .stat-card {{
    flex: 1; background: white; border-radius: 10px; padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border-left: 4px solid #1f77b4;
  }}
  .stat-card.green {{ border-left-color: #28a745; }}
  .stat-card.red   {{ border-left-color: #dc3545; }}
  .stat-card .val {{ font-size: 1.8rem; font-weight: 700; color: #333; }}
  .stat-card .lbl {{ font-size: 0.8rem; color: #888; margin-top: 2px; }}

  table {{ width: 100%; border-collapse: collapse; background: white;
           border-radius: 10px; overflow: hidden;
           box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  th {{ background: #343a40; color: white; padding: 12px 14px; text-align: left;
        font-weight: 600; font-size: 0.85rem; }}
  th.idx {{ width: 50px; text-align: center; }}
  th.label {{ width: 100px; text-align: center; }}
  th.status {{ width: 60px; text-align: center; }}
  td {{ padding: 10px 14px; font-size: 0.88rem; border-bottom: 1px solid #eee; }}
  td.idx {{ text-align: center; color: #999; font-size: 0.8rem; }}
  td.text {{ max-width: 500px; font-family: 'Georgia', serif; color: #444; }}
  td.label {{ text-align: center; font-weight: 600; border-radius: 4px; padding: 4px 8px; }}
  td.status {{ text-align: center; font-size: 1rem; }}

  .legend {{
    display: flex; gap: 20px; margin-bottom: 16px; font-size: 0.85rem; color: #666;
  }}
  .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
  .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}

  .label-neutral {{ background: {LABEL_COLORS['neutral']}; color: white; padding: 2px 8px; border-radius: 4px; }}
  .label-positive {{ background: {LABEL_COLORS['positive']}; color: white; padding: 2px 8px; border-radius: 4px; }}
  .label-negative {{ background: {LABEL_COLORS['negative']}; color: white; padding: 2px 8px; border-radius: 4px; }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>Prediction Comparison — {exp['name']}</h1>
    <div class="meta">
      <div>Model: {exp['model'].upper()}</div>
      <div>lr={exp['lr']}  bs={exp['bs']}</div>
      <div>Top-{n_rows} test samples</div>
    </div>
  </div>

  <div class="stats">
    <div class="stat-card green">
      <div class="val">{acc:.1f}%</div>
      <div class="lbl">Accuracy ({n_rows} samples)</div>
    </div>
    <div class="stat-card">
      <div class="val">{(df[true_col]==df[pred_col]).sum()}</div>
      <div class="lbl">Correct</div>
    </div>
    <div class="stat-card red">
      <div class="val">{(df[true_col]!=df[pred_col]).sum()}</div>
      <div class="lbl">Incorrect</div>
    </div>
  </div>

  <div class="legend">
    <span><span class="dot" style="background:{LABEL_COLORS['neutral']}"></span> neutral</span>
    <span><span class="dot" style="background:{LABEL_COLORS['positive']}"></span> positive</span>
    <span><span class="dot" style="background:{LABEL_COLORS['negative']}"></span> negative</span>
    <span><span style="color:#28a745">&#10004;</span> Correct</span>
    <span><span style="color:#dc3545">&#10008;</span> Incorrect</span>
  </div>

  <table>
    <thead>
      <tr>
        <th class="idx">#</th>
        <th>Text</th>
        <th class="label">True Label</th>
        <th class="label">Pred Label</th>
        <th class="status">OK</th>
      </tr>
    </thead>
    <tbody>
{rows_str}
    </tbody>
  </table>

</div>
</body>
</html>"""

        out_html = os.path.join(output_dir, f'prediction_table_{exp["name"]}.html')
        with open(out_html, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f'  [Saved] {out_html}')

    # 同时生成一张汇总图片（所有实验的预测分布热力图）
    _plot_prediction_heatmap(experiments, output_dir)


def _esc(s):
    """HTML 转义"""
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
             .replace('"', '&quot;').replace("'", '&#39;'))


def _plot_prediction_heatmap(experiments, output_dir):
    """
    生成预测分布对比热力图（图片版），作为 HTML 表格的补充。
    """
    set_style()
    labels = ['neutral', 'positive', 'negative']

    fig, axes = plt.subplots(1, len(experiments), figsize=(5 * max(1, len(experiments)), 4), squeeze=False)
    fig.suptitle('Prediction Distribution per Experiment', fontsize=13, fontweight='bold')

    for col, exp in enumerate(experiments):
        pred_df = exp['predictions']
        if pred_df is None:
            continue

        true_col  = 'true_label'
        pred_col = 'pred_label'

        # 混淆矩阵风格：行=true, 列=pred
        matrix = np.zeros((3, 3), dtype=int)
        label_id = {'neutral': 0, 'positive': 1, 'negative': 2,
                    '0': 0, '1': 1, '2': 2,
                    0: 0, 1: 1, 2: 2}
        for _, row in pred_df.iterrows():
            t = label_id.get(row[true_col], 0)
            p = label_id.get(row[pred_col], 0)
            matrix[t, p] += 1

        ax = axes[0, col]
        im = ax.imshow(matrix, cmap='Blues', aspect='auto')

        ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True Label')
        ax.set_title(exp['name'], fontsize=10)

        for r in range(3):
            for c in range(3):
                val = matrix[r, c]
                color = 'white' if val > matrix.max() * 0.5 else 'black'
                ax.text(c, r, str(val), ha='center', va='center', color=color, fontsize=11, fontweight='bold')

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'prediction_heatmap.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  [Saved] {out_path}')


# ==================== 汇总报告 ====================

def generate_summary(experiments, output_dir):
    """生成文本摘要表（CSV + 控制台）"""
    rows = []
    for exp in experiments:
        df = exp['metrics']
        if df.empty:
            continue
        best_val_acc = df['val_acc'].max()
        best_epoch   = df.loc[df['val_acc'].idxmax(), 'epoch']
        final_val_acc = df['val_acc'].iloc[-1]
        final_val_loss = df['val_loss'].iloc[-1]
        pred_df = exp['predictions']
        test_acc = (pred_df['true_label'] == pred_df['pred_label']).mean() * 100 if pred_df is not None else float('nan')

        rows.append({
            'experiment':  exp['name'],
            'model':       exp['model'],
            'lr':          exp['lr'],
            'bs':          exp['bs'],
            'best_epoch':  int(best_epoch),
            'best_val_acc': f'{best_val_acc:.4f}',
            'final_val_acc': f'{final_val_acc:.4f}',
            'final_val_loss': f'{final_val_loss:.4f}',
            'test_acc(%)':  f'{test_acc:.2f}',
        })

    summary_df = pd.DataFrame(rows)
    out_csv = os.path.join(output_dir, 'experiment_summary.csv')
    summary_df.to_csv(out_csv, index=False)
    print(f'\n  [Saved] {out_csv}')

    print('\n  ===== Experiment Summary =====')
    print(summary_df.to_string(index=False))
    return summary_df


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description='绘制实验结果曲线和预测对比表')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'results'),
                        help='实验结果根目录')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认: results_dir）')
    parser.add_argument('--n_rows', type=int, default=100,
                        help='HTML 表格展示的行数（默认 100）')
    parser.add_argument('--separate', action='store_true',
                        help='同时生成 6 张独立图片')
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    output_dir  = os.path.abspath(args.output_dir) if args.output_dir else results_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f'\nScanning: {results_dir}')
    experiments = discover_experiments(results_dir)

    if not experiments:
        print('[Error] No experiment results found.')
        print('  Expected structure: results/<exp_name>/training_metrics.csv')
        print('  Or: results/training_metrics.csv (single experiment)')
        return

    print(f'\nFound {len(experiments)} experiment(s):')
    for e in experiments:
        print(f'  - {e["name"]}  model={e["model"]}  lr={e["lr"]}  bs={e["bs"]}')

    # 1. 6合1曲线图
    print('\n[1/3] Drawing training curves (6-in-1)...')
    plot_all_curves(experiments, output_dir, title_prefix=' | '.join(set(e['model'] for e in experiments)) + ' — ')

    # 2. 6张独立图片
    if args.separate:
        print('\n[2/3] Drawing separate figures...')
        plot_separate_figures(experiments, output_dir)

    # 3. HTML 预测表格 + 热力图
    print('\n[3/3] Generating HTML prediction tables...')
    generate_html_table(experiments, output_dir, n_rows=args.n_rows)

    # 4. 汇总报告
    print('\n[4/3] Generating summary report...')
    generate_summary(experiments, output_dir)

    print(f'\nAll outputs saved to: {output_dir}')
    print('Done.')


if __name__ == '__main__':
    main()
