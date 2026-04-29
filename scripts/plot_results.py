"""
实验结果可视化脚本
自动扫描 results/ 目录下的所有实验结果，绘制训练曲线，
并生成精美的 HTML 预测对比表格。

Results 目录结构：
    results/
    ├── baseline_loss_crossentropy_lr_0p001_bs_32/
    │   ├── training_metrics.csv   (epoch, train_loss, train_acc, val_loss, val_acc)
    │   ├── experiment_summary.json
    │   └── prediction_comparison.csv  (text, true_label, pred_label, correct)
    ├── finbert_loss_crossentropy_lr_2e-5_bs_16/
    │   └── ...
    └── ...

用法：
    python scripts/plot_results.py
    python scripts/plot_results.py --results_dir results --output_dir results/plots
"""

import os
import re
import json
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== 颜色 / 样式配置 ====================

# 每个 (model, loss_fn, lr, bs) 组合对应一个样式
MARKER_STYLE_MAP = [
    {'color': '#1f77b4', 'ls': '-',  'marker': 'o'},
    {'color': '#ff7f0e', 'ls': '--', 'marker': 's'},
    {'color': '#2ca02c', 'ls': '-.', 'marker': '^'},
    {'color': '#d62728', 'ls': ':',  'marker': 'D'},
    {'color': '#9467bd', 'ls': '-',  'marker': 'v'},
    {'color': '#8c564b', 'ls': '--', 'marker': 'P'},
    {'color': '#e377c2', 'ls': '-.', 'marker': 'X'},
    {'color': '#7f7f7f', 'ls': ':',  'marker': 'h'},
]


# ==================== 目录扫描 ====================

def discover_experiments(results_dir):
    """
    扫描 results_dir 下所有包含 training_metrics.csv 的子目录。
    Returns:
        List[dict]  每个 dict 包含: path, name, model, loss_fn, lr, bs, metrics_df, pred_df
    """
    experiments = []
    pattern = os.path.join(results_dir, '*', 'training_metrics.csv')
    for csv_path in glob.glob(pattern):
        exp_dir = os.path.dirname(csv_path)
        exp_name = os.path.basename(exp_dir)
        model, loss_fn, lr, bs = _parse_exp_name(exp_name)

        try:
            metrics_df = pd.read_csv(csv_path)
        except Exception:
            continue

        abs_exp_dir = os.path.abspath(exp_dir)

        pred_path = os.path.join(abs_exp_dir, 'prediction_comparison.csv')
        pred_df = pd.read_csv(pred_path) if os.path.exists(pred_path) else None

        # 从 summary json 读取所有字段（目录名只作为回退）
        json_path = os.path.join(abs_exp_dir, 'experiment_summary.json')
        model_from_json = loss_fn_from_json = lr_from_json = bs_from_json = ''
        if os.path.exists(json_path):
            try:
                with open(json_path) as fj:
                    summary = json.load(fj)
                    model_from_json   = str(summary.get('model', ''))
                    loss_fn_from_json = str(summary.get('loss_fn', ''))
                    lr_from_json     = str(summary.get('lr', ''))
                    bs_from_json     = str(summary.get('batch_size', ''))
            except Exception:
                pass

        experiments.append({
            'name':     exp_name,
            'model':    model_from_json if model_from_json else (model or 'unknown'),
            'loss_fn':  loss_fn_from_json if loss_fn_from_json else (loss_fn or 'crossentropy'),
            'lr':       lr_from_json if lr_from_json else lr,
            'bs':       bs_from_json if bs_from_json else bs,
            'path':     abs_exp_dir,
            'metrics':  metrics_df,
            'predictions': pred_df,
        })

    # 排序：模型 -> 损失函数 -> lr -> bs
    def sort_key(e):
        # 优先使用 JSON 中的 lr 数值排序（目录名 lr 可能有歧义）
        try:
            lr_val = float(e['lr'])  # lr_from_json 通常是干净的浮点数字符串
        except Exception:
            try:
                lr_val = float(str(e['lr']).replace('p', '.'))
            except Exception:
                lr_val = 0
        bs_val = int(e['bs']) if e['bs'] else 0
        return (e['model'], e['loss_fn'], lr_val, bs_val)

    experiments.sort(key=sort_key)
    return experiments


def _parse_exp_name(name):
    """
    从目录名解析 model / loss_fn / lr / bs
    格式: {model}_loss_{loss_fn}_lr_{lr}_bs_{bs}
    例如: baseline_loss_crossentropy_lr_0p001_bs_32
          finbert_loss_crossentropy_lr_2e-5_bs_16
    """
    m = re.match(
        r'([a-zA-Z_]+)_loss_([a-zA-Z_]+)_lr_([0-9pe.\-]+)_bs_(\d+)',
        name, re.I
    )
    if m:
        # 还原 lr 格式: 0p001 -> 0.001, 1e-3 -> 0.001
        lr_raw = m.group(3)
        lr_str = lr_raw.replace('p', '.')
        return m.group(1).lower(), m.group(2).lower(), lr_str, m.group(4)
    return None, None, None, None


# ==================== 样式 ====================

def _get_style(idx):
    return MARKER_STYLE_MAP[idx % len(MARKER_STYLE_MAP)]


def _fmt_lr(lr_str):
    """将 lr 转为可读字符串（如 '0.001' -> '0.001', '0.01' -> '0.01'）"""
    try:
        v = float(lr_str)
        # 极小值（< 1e-4）用科学计数法，其余保留合理小数位
        if abs(v) < 1e-4:
            return f'{v:.0e}'  # e.g. 0.0001 -> '1e-4'
        s = f'{v:.4f}'.rstrip('0').rstrip('.')  # e.g. 0.01 -> '0.01', 0.1 -> '0.1', 0.001 -> '0.001'
        return s
    except Exception:
        # 回退处理目录名格式（如 '0p001'）
        try:
            v = float(lr_str.replace('p', '.'))
            return f'{v:.4f}'.rstrip('0').rstrip('.')
        except Exception:
            return lr_str


# ==================== Matplotlib 全局样式 ====================

def _setup_matplotlib():
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
        'grid.linestyle': '--',
    })


# ==================== 绘图函数 ====================

def plot_combined_curves(experiments, output_dir):
    """
    绘制 6 合 1 大图：
        Row 1: BiLSTM Loss | FinBERT Loss | All Models Val Loss
        Row 2: BiLSTM Acc | FinBERT Acc  | All Models Val Acc
    """
    _setup_matplotlib()

    bilstm = [e for e in experiments
              if 'baseline' in e['model'].lower()
              or 'bilstm' in e['model'].lower()
              or 'bilstm' in e['name'].lower()]
    finbert = [e for e in experiments if 'bert' in e['model'].lower() or 'bert' in e['name'].lower()]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training & Validation Curves', fontsize=16, fontweight='bold', y=1.01)

    _draw_loss_or_acc(
        axes[0, 0], bilstm,
        train_col='train_loss', val_col='val_loss',
        ylabel='Loss', title='BiLSTM — Training & Validation Loss',
        show_train=True,
    )
    _draw_loss_or_acc(
        axes[0, 1], finbert,
        train_col='train_loss', val_col='val_loss',
        ylabel='Loss', title='FinBERT — Training & Validation Loss',
        show_train=True,
    )
    _draw_loss_or_acc(
        axes[0, 2], experiments,
        train_col='train_loss', val_col='val_loss',
        ylabel='Loss', title='All Models — Validation Loss',
        show_train=False,
    )
    _draw_loss_or_acc(
        axes[1, 0], bilstm,
        train_col='train_acc', val_col='val_acc',
        ylabel='Accuracy', title='BiLSTM — Training & Validation Accuracy',
        show_train=True,
    )
    _draw_loss_or_acc(
        axes[1, 1], finbert,
        train_col='train_acc', val_col='val_acc',
        ylabel='Accuracy', title='FinBERT — Training & Validation Accuracy',
        show_train=True,
    )
    _draw_loss_or_acc(
        axes[1, 2], experiments,
        train_col='train_acc', val_col='val_acc',
        ylabel='Accuracy', title='All Models — Validation Accuracy',
        show_train=False,
    )

    plt.tight_layout()
    out_png = os.path.join(output_dir, 'training_curves_combined.png')
    out_pdf = os.path.join(output_dir, 'training_curves_combined.pdf')
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f'  [Saved] {out_png}')
    print(f'  [Saved] {out_pdf}')
    return out_png


def plot_separate_figures(experiments, output_dir):
    """绘制 6 张独立图片"""
    _setup_matplotlib()

    bilstm  = [e for e in experiments
               if 'baseline' in e['model'].lower()
               or 'bilstm' in e['model'].lower()
               or 'bilstm' in e['name'].lower()]
    finbert = [e for e in experiments if 'bert' in e['model'].lower() or 'bert' in e['name'].lower()]

    configs = [
        ('fig01_bilstm_loss.png',    bilstm,           'train_loss', 'val_loss',  'Loss',     'BiLSTM — Training & Validation Loss',              True),
        ('fig02_finbert_loss.png',   finbert,          'train_loss', 'val_loss',  'Loss',     'FinBERT — Training & Validation Loss (varying batch sizes)', True),
        ('fig03_loss_comparison.png', experiments,      'train_loss', 'val_loss',  'Loss',     'All Models — Validation Loss Comparison',          False),
        ('fig04_bilstm_acc.png',     bilstm,           'train_acc',  'val_acc',   'Accuracy', 'BiLSTM — Training & Validation Accuracy',          True),
        ('fig05_finbert_acc.png',    finbert,          'train_acc',  'val_acc',   'Accuracy', 'FinBERT — Training & Validation Accuracy (varying batch sizes)', True),
        ('fig06_acc_comparison.png',  experiments,      'train_acc',  'val_acc',   'Accuracy', 'All Models — Validation Accuracy Comparison',    False),
    ]

    for filename, exps, tc, vc, ylabel, title, show_train in configs:
        fig, ax = plt.subplots(figsize=(10, 6.5) if show_train else (10, 6))
        _draw_loss_or_acc(ax, exps, tc, vc, ylabel, title, show_train=show_train)
        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f'  [Saved] {out_path}')


def _draw_loss_or_acc(ax, experiments, train_col, val_col, ylabel, title, show_train=True):
    """在单个 ax 上绘制曲线"""
    if not experiments:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=13, color='#999')
        ax.set_title(title); ax.set_ylabel(ylabel)
        return

    for i, exp in enumerate(experiments):
        df = exp['metrics']
        if train_col not in df.columns or val_col not in df.columns:
            continue

        epochs = df['epoch'].values
        lr_str = _fmt_lr(exp['lr'])
        label = f"lr={lr_str}, bs={exp['bs']}, loss={exp['loss_fn']}"
        s = _get_style(i)

        if show_train:
            ax.plot(epochs, df[train_col].values,
                    linestyle='--', marker=s['marker'], markersize=4, linewidth=1.4,
                    color=s['color'], alpha=0.55,
                    label=f'{label} (train)')
            ax.plot(epochs, df[val_col].values,
                    linestyle=s['ls'], marker=s['marker'], markersize=5, linewidth=2,
                    color=s['color'],
                    label=f'{label} (val)')
        else:
            ax.plot(epochs, df[val_col].values,
                    linestyle=s['ls'], marker=s['marker'], markersize=5, linewidth=2,
                    color=s['color'],
                    label=label)

    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    if show_train:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2,
                  framealpha=0.9, fontsize=8)
    else:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), framealpha=0.9, fontsize=8)
    ax.set_xlim(left=1)
    ax.set_xticks(range(1, int(ax.get_xlim()[1]) + 1))


# ==================== HTML 预测表格 ====================

LABEL_BG = {
    'neutral':  '#6c757d',
    'positive': '#28a745',
    'negative': '#dc3545',
    '0': '#6c757d',
    '1': '#28a745',
    '2': '#dc3545',
}


def _esc(s):
    s = str(s)
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
             .replace('"', '&quot;').replace("\n", ' ').replace("\r", ''))


def generate_html_tables(experiments, output_dir, n_rows=100):
    """为每个实验生成一个精美的 HTML 预测对比表格"""
    os.makedirs(output_dir, exist_ok=True)

    for exp in experiments:
        pred_df = exp['predictions']
        if pred_df is None or pred_df.empty:
            continue

        n_rows = min(n_rows, len(pred_df))
        df = pred_df.head(n_rows).reset_index(drop=True)

        true_col = 'true_label'
        pred_col = 'pred_label'

        rows_html = []
        for i, row in df.iterrows():
            idx   = i + 1
            text  = str(row.get('text', ''))
            true_l = str(row.get(true_col, ''))
            pred_l = str(row.get(pred_col, ''))
            is_correct = str(row.get('correct', '')).lower() in ('true', '1', 'yes', 'true')
            row_bg = '#f8f9fa' if i % 2 == 0 else '#ffffff'
            icon = '&#10004;' if is_correct else '&#10008;'
            icon_color = '#28a745' if is_correct else '#dc3545'
            true_bg = LABEL_BG.get(true_l, '#343a40')
            pred_bg = LABEL_BG.get(pred_l, '#343a40') if not is_correct else '#28a745'
            text_disp = (_esc(text[:150]) + '...') if len(text) > 150 else _esc(text)

            rows_html.append(f"""
            <tr style="background:{row_bg};">
              <td class="idx">{idx}</td>
              <td class="text" title="{_esc(text)}">{text_disp}</td>
              <td class="label" style="background:{true_bg};">{true_l}</td>
              <td class="label" style="background:{pred_bg};">{pred_l}</td>
              <td class="ok" style="color:{icon_color};">{icon}</td>
            </tr>""")

        rows_str = '\n'.join(rows_html)
        acc = (df[true_col] == df[pred_col]).mean() * 100
        n_correct = int((df[true_col] == df[pred_col]).sum())
        n_total = len(df)
        n_wrong = n_total - n_correct

        html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Prediction Comparison — {exp['name']}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
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
    text-align: right; font-size: 0.85rem; line-height: 1.8;
  }}
  .stats {{ display: flex; gap: 12px; margin-bottom: 20px; }}
  .card {{ flex: 1; background: white; border-radius: 10px; padding: 16px 20px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
  .card .v {{ font-size: 1.8rem; font-weight: 700; color: #333; }}
  .card .l {{ font-size: 0.75rem; color: #888; margin-top: 2px; }}
  .card.green {{ border-top: 3px solid #28a745; }}
  .card.blue  {{ border-top: 3px solid #1f77b4; }}
  .card.red   {{ border-top: 3px solid #dc3545; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
           border-radius: 10px; overflow: hidden;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th {{ background: #343a40; color: white; padding: 11px 14px;
        text-align: left; font-weight: 600; font-size: 0.82rem; }}
  th.idx {{ width: 45px; text-align: center; }}
  th.label {{ width: 100px; text-align: center; }}
  th.ok {{ width: 50px; text-align: center; }}
  td {{ padding: 9px 14px; font-size: 0.87rem; border-bottom: 1px solid #eee; }}
  td.idx {{ text-align: center; color: #bbb; font-size: 0.78rem; }}
  td.text {{ max-width: 480px; font-family: 'Georgia', serif; color: #444; }}
  td.label {{ text-align: center; font-weight: 700; border-radius: 4px;
              padding: 3px 8px; color: white; font-size: 0.82rem; }}
  td.ok {{ text-align: center; font-size: 1rem; }}
  .legend {{ display: flex; gap: 18px; margin-bottom: 14px; font-size: 0.82rem; color: #666; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 5px; }}
  .dot {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>Prediction Comparison — {exp['name']}</h1>
    <div class="meta">
      <div>Model: {exp['model'].upper()}</div>
      <div>lr={_fmt_lr(exp['lr'])}, bs={exp['bs']}</div>
      <div>loss_fn={exp['loss_fn']}</div>
    </div>
  </div>
  <div class="stats">
    <div class="card green"><div class="v">{acc:.1f}%</div><div class="l">Accuracy ({n_rows} samples)</div></div>
    <div class="card blue"><div class="v">{n_correct}</div><div class="l">Correct</div></div>
    <div class="card red"><div class="v">{n_wrong}</div><div class="l">Incorrect</div></div>
  </div>
  <div class="legend">
    <span><span class="dot" style="background:{LABEL_BG['neutral']}"></span>neutral</span>
    <span><span class="dot" style="background:{LABEL_BG['positive']}"></span>positive</span>
    <span><span class="dot" style="background:{LABEL_BG['negative']}"></span>negative</span>
    <span><span style="color:#28a745">&#10004;</span>Correct</span>
    <span><span style="color:#dc3545">&#10008;</span>Incorrect</span>
  </div>
  <table>
    <thead>
      <tr><th class="idx">#</th><th>Text</th><th class="label">True Label</th><th class="label">Pred Label</th><th class="ok">OK</th></tr>
    </thead>
    <tbody>{rows_str}
    </tbody>
  </table>
</div>
</body>
</html>"""

        out_html = os.path.join(output_dir, f'prediction_table_{exp["name"]}.html')
        with open(out_html, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f'  [Saved] {out_html}')

    # 生成预测分布热力图
    _plot_heatmap(experiments, output_dir)


def _build_matrix(pred_df):
    """从预测 DataFrame 构建 3x3 混淆矩阵"""
    label_id = {'neutral': 0, 'positive': 1, 'negative': 2,
                '0': 0, '1': 1, '2': 2, 0: 0, 1: 1, 2: 2}
    matrix = np.zeros((3, 3), dtype=int)
    for _, row in pred_df.iterrows():
        t = label_id.get(row['true_label'], 0)
        p = label_id.get(row['pred_label'], 0)
        matrix[t, p] += 1
    return matrix


def _draw_one_heatmap(ax, matrix, title, labels):
    """在单个 ax 上绘制一张混淆矩阵"""
    im = ax.imshow(matrix, cmap='Blues', aspect='auto')
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True Label')
    ax.set_title(title, fontsize=10)
    for r in range(3):
        for c in range(3):
            color = 'white' if matrix[r, c] > matrix.max() * 0.5 else 'black'
            ax.text(c, r, str(matrix[r, c]), ha='center', va='center',
                    color=color, fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)


def _plot_heatmap(experiments, output_dir):
    """生成三张混淆矩阵热力图：BiLSTM-CE / BiLSTM-LS (各3 LR) / FinBERT (最佳3个)"""
    _setup_matplotlib()
    labels = ['neutral', 'positive', 'negative']

    def _lr_match(lr_str, targets=(0.01, 0.001, 0.0001)):
        try:
            return round(float(lr_str), 5) in targets
        except Exception:
            return False

    def _is_ls(exp):
        return 'smoothing' in exp['loss_fn'].lower()

    def _is_bilstm(exp):
        m = exp['model'].lower()
        return 'baseline' in m or 'bilstm' in m

    def _best_exps(exps, n=3):
        """返回 val_acc 最高的 n 个实验"""
        if not exps:
            return []
        sorted_exps = sorted(exps, key=lambda e: e['metrics']['val_acc'].max() if not e['metrics'].empty else 0, reverse=True)
        return sorted_exps[:n]

    bilstm_ce = [e for e in experiments
                 if _is_bilstm(e) and not _is_ls(e) and _lr_match(e['lr'])]
    bilstm_ls = [e for e in experiments
                 if _is_bilstm(e) and _is_ls(e) and _lr_match(e['lr'])]
    finbert_best3 = _best_exps([e for e in experiments if 'bert' in e['model'].lower()], n=3)

    # 图1: BiLSTM CrossEntropy (3 LR)
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
    fig1.suptitle('BiLSTM + Attention (CrossEntropy Loss) — Confusion Matrix (Test Set)', fontsize=13, fontweight='bold')
    for i, exp in enumerate(bilstm_ce):
        ax = axes1[0, i]
        m = _build_matrix(exp['predictions']) if exp['predictions'] is not None else np.zeros((3,3))
        _draw_one_heatmap(ax, m, exp['name'], labels)
    plt.tight_layout()
    out1 = os.path.join(output_dir, 'fig08_heatmap_bilstm_ce.png')
    fig1.savefig(out1, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  [Saved] {out1}')

    # 图2: BiLSTM Label Smoothing (3 LR)
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
    fig2.suptitle('BiLSTM + Attention (Label Smoothing Loss) — Confusion Matrix (Test Set)', fontsize=13, fontweight='bold')
    for i, exp in enumerate(bilstm_ls):
        ax = axes2[0, i]
        m = _build_matrix(exp['predictions']) if exp['predictions'] is not None else np.zeros((3,3))
        _draw_one_heatmap(ax, m, exp['name'], labels)
    plt.tight_layout()
    out2 = os.path.join(output_dir, 'fig09_heatmap_bilstm_ls.png')
    fig2.savefig(out2, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  [Saved] {out2}')

    # 图3: FinBERT (最佳3个)
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
    fig3.suptitle('FinBERT — Confusion Matrix (Test Set, Best 3 by Val Accuracy)', fontsize=13, fontweight='bold')
    for i, exp in enumerate(finbert_best3):
        ax = axes3[0, i]
        m = _build_matrix(exp['predictions']) if exp['predictions'] is not None else np.zeros((3,3))
        _draw_one_heatmap(ax, m, exp['name'], labels)
    plt.tight_layout()
    out3 = os.path.join(output_dir, 'fig10_heatmap_finbert.png')
    fig3.savefig(out3, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  [Saved] {out3}')


# ==================== 汇总报告 ====================

def generate_summary_csv(experiments, output_dir):
    """生成 experiment_summary_all.csv 汇总表"""
    rows = []
    for exp in experiments:
        df = exp['metrics']
        if df.empty:
            continue
        best_idx = df['val_acc'].idxmax()
        pred_df = exp['predictions']
        test_acc = (
            (pred_df['true_label'] == pred_df['pred_label']).mean() * 100
            if pred_df is not None else float('nan')
        )
        rows.append({
            'experiment':       exp['name'],
            'model':            exp['model'],
            'loss_fn':          exp['loss_fn'],
            'lr':               _fmt_lr(exp['lr']),
            'batch_size':       exp['bs'],
            'best_epoch':       int(df.loc[best_idx, 'epoch']),
            'best_val_acc':     round(float(df.loc[best_idx, 'val_acc']), 4),
            'final_val_acc':    round(float(df['val_acc'].iloc[-1]), 4),
            'final_val_loss':   round(float(df['val_loss'].iloc[-1]), 4),
            'test_acc(%)':      round(test_acc, 2),
        })

    summary_df = pd.DataFrame(rows)
    out_csv = os.path.join(output_dir, 'experiment_summary_all.csv')
    summary_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f'  [Saved] {out_csv}')

    print('\n  ===== Experiment Summary =====')
    try:
        print(summary_df.to_string(index=False))
    except Exception:
        for _, row in summary_df.iterrows():
            print(dict(row))
    return summary_df


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description='实验结果可视化')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'results'),
                        help='实验结果根目录')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认同 results_dir）')
    parser.add_argument('--n_rows', type=int, default=100,
                        help='HTML 表格展示行数（默认 100）')
    # always separate; combined view removed
    # (kept for backwards compat — no-op)
    parser.add_argument('--separate', action='store_true',
                        help='[deprecated, now always on] 同时生成 6 张独立图片')
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    output_dir  = os.path.abspath(args.output_dir) if args.output_dir else results_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f'\nScanning: {results_dir}')
    experiments = discover_experiments(results_dir)

    if not experiments:
        print('[Error] No experiment results found.')
        print('  Expected: results/<exp_name>/training_metrics.csv')
        return

    print(f'\nFound {len(experiments)} experiment(s):')
    for e in experiments:
        print(f'  - {e["name"]}  model={e["model"]}  loss={e["loss_fn"]}  '
              f'lr={_fmt_lr(e["lr"])}  bs={e["bs"]}')

    print('\n[1/4] Drawing separate figures...')
    plot_separate_figures(experiments, output_dir)

    print('\n[3/4] Generating HTML prediction tables...')
    generate_html_tables(experiments, output_dir, n_rows=args.n_rows)

    print('\n[4/4] Generating summary report...')
    generate_summary_csv(experiments, output_dir)

    print(f'\nAll outputs saved to: {output_dir}')
    print('Done.')


if __name__ == '__main__':
    main()
