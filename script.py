from pathlib import Path
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', context='talk')

CSV_PATH = Path('notas_manuales.csv')
OUTPUT_DIR = Path('output/parte4_notebook')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FAMILY_POSITIONS = {
    'DENSO': [3, 6, 9, 12],
    'GRAPH': [3, 6, 9, 12],
    'SPARSE': [3, 6, 9, 12, 15, 18, 21, 24, 27],
    'RFF': [3, 6, 9, 12, 15, 18, 21, 24, 27],
}

FAMILY_LABELS = {
    'DENSO': 'DENSO [COS]',
    'GRAPH': 'GRAPH',
    'SPARSE': 'SPARSE',
    'RFF': 'RRF [HNSW + SPARSE]',
}

QUESTION_LABELS = [f'P{i}' for i in range(1, 10)]


def detect_family(label: str) -> str | None:
    if label.startswith('DENSO'):
        return 'DENSO'
    if label.startswith('GRAPH'):
        return 'GRAPH'
    if label.startswith('SPARSE'):
        return 'SPARSE'
    if label.startswith('RFF'):
        return 'RFF'
    return None


records = []
with CSV_PATH.open(encoding='utf-8-sig', newline='') as file:
    reader = csv.reader(file)
    next(reader, None)
    next(reader, None)
    for row in reader:
        if len(row) < 2:
            continue
        label = row[1].strip()
        family = detect_family(label)
        if family is None:
            continue

        positions = FAMILY_POSITIONS[family]
        for question_index, score_pos in enumerate(positions, start=1):
            if score_pos >= len(row):
                continue
            raw_score = row[score_pos].strip()
            if not raw_score:
                continue
            try:
                score = float(raw_score.replace(',', '.'))
            except ValueError:
                continue

            similarity = None
            if score_pos + 1 < len(row) and row[score_pos + 1].strip():
                try:
                    similarity = float(row[score_pos + 1].replace(',', '.'))
                except ValueError:
                    similarity = None

            doc_id = row[score_pos + 2].strip() if score_pos + 2 < len(row) else ''
            records.append(
                {
                    'family': family,
                    'method_label': label,
                    'question': question_index,
                    'score': score,
                    'similarity': similarity,
                    'doc_id': doc_id,
                }
            )


df = pd.DataFrame(records)
score_summary = (
    df.groupby(['family', 'question'], as_index=False)['score']
    .mean()
    .sort_values(['family', 'question'])
)
coverage = (
    df.assign(present=1)
    .pivot_table(index='family', columns='question', values='present', aggfunc='sum', fill_value=0)
    .reindex(['DENSO', 'GRAPH', 'SPARSE', 'RFF'])
)
score_pivot = score_summary.pivot(index='family', columns='question', values='score')
method_global = (
    df.groupby('family', as_index=False)['score']
    .mean()
    .rename(columns={'score': 'mean_score'})
)

score_summary

from IPython.display import display

summary_table = pd.DataFrame(index=['DENSO [COS]', 'GRAPH', 'SPARSE', 'RRF [HNSW + SPARSE]'], columns=[f'p{i}' for i in range(1, 10)], dtype=object)

for family, label in FAMILY_LABELS.items():
    family_scores = score_summary[score_summary['family'] == family].set_index('question')['score']
    for question in range(1, 10):
        value = family_scores.get(question)
        if pd.notna(value):
            summary_table.loc[label, f'p{question}'] = round(float(value), 2)

summary_table

def format_cell(value):
    if value is None or pd.isna(value):
        return ''
    if isinstance(value, str) and not value.strip():
        return ''
    return f'{float(value):.2f}'

latex_lines = [
    '\\begin{table}[h]',
    '\\centering',
    '\\small',
    '\\renewcommand{\\arraystretch}{1.15}',
    '\\setlength{\\tabcolsep}{6pt}',
    '\\caption{Resumen de evaluación manual por método y pregunta}',
    '\\label{tab:resumen-parte4}',
    '\\resizebox{\\textwidth}{!}{%',
    '\\begin{tabular}{|l|' + 'c|' * 9 + '}',
    '\\hline',
    '\\textbf{Método} & \\textbf{$\\bar{X}_{P1}$} & \\textbf{$\\bar{X}_{P2}$} & \\textbf{$\\bar{X}_{P3}$} & \\textbf{$\\bar{X}_{P4}$} & \\textbf{$\\bar{X}_{P5}$} & \\textbf{$\\bar{X}_{P6}$} & \\textbf{$\\bar{X}_{P7}$} & \\textbf{$\\bar{X}_{P8}$} & \\textbf{$\\bar{X}_{P9}$} \\\\',
    '\\hline',
]

for method in summary_table.index:
    values = ' & '.join(format_cell(summary_table.loc[method, f'p{i}']) for i in range(1, 10))
    latex_lines.append(f'{method} & {values} \\\\')
    latex_lines.append('\\hline')

latex_lines.extend([
    '\\end{tabular}}',
    '\\end{table}',
])

latex_table = '\n'.join(latex_lines)
latex_output_path = OUTPUT_DIR / 'tabla_resumen_parte4.tex'
latex_output_path.write_text(latex_table, encoding='utf-8')
latex_table
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Evaluación manual - Parte 4', fontsize=20, fontweight='bold')

# 1) Comparación p1-p4 entre todos los métodos
plot_p14 = score_summary[score_summary['question'].isin([1, 2, 3, 4])].copy()
plot_p14['question_label'] = plot_p14['question'].map(lambda x: f'P{x}')
plot_p14['family_label'] = plot_p14['family'].map(FAMILY_LABELS)
sns.barplot(data=plot_p14, x='question_label', y='score', hue='family_label', ax=axes[0, 0])
axes[0, 0].set_title('Promedio manual por pregunta (P1-P4)')
axes[0, 0].set_xlabel('Pregunta')
axes[0, 0].set_ylabel('Promedio')
axes[0, 0].legend(title='Método', fontsize=9)

# 2) Promedio global por método
method_global_plot = method_global.copy()
method_global_plot['family_label'] = method_global_plot['family'].map(FAMILY_LABELS)
sns.barplot(data=method_global_plot, x='family_label', y='mean_score', ax=axes[0, 1], palette='viridis')
axes[0, 1].set_title('Promedio global por método')
axes[0, 1].set_xlabel('Método')
axes[0, 1].set_ylabel('Promedio')
axes[0, 1].tick_params(axis='x', rotation=15)

# 3) SPARSE vs RRF en todas las preguntas
sparse_rff = score_summary[score_summary['family'].isin(['SPARSE', 'RFF'])].copy()
sparse_rff['family_label'] = sparse_rff['family'].map(FAMILY_LABELS)
sns.lineplot(data=sparse_rff, x='question', y='score', hue='family_label', marker='o', ax=axes[1, 0])
axes[1, 0].set_title('SPARSE vs RRF en todas las preguntas')
axes[1, 0].set_xlabel('Pregunta')
axes[1, 0].set_ylabel('Promedio')
axes[1, 0].set_xticks(range(1, 10))
axes[1, 0].legend(title='Método')



plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUTPUT_DIR / 'comparativa_general.png', dpi=180, bbox_inches='tight')
plt.show()

# 5) Diferencia RRF - SPARSE por pregunta
sparse_pivot = sparse_rff.pivot(index='question', columns='family', values='score')
delta = (sparse_pivot['RFF'] - sparse_pivot['SPARSE']).dropna()
plt.figure(figsize=(12, 5))
sns.barplot(x=delta.index, y=delta.values, color='#6B7FD7')
plt.axhline(0, color='black', linewidth=1)
plt.title('Diferencia RRF - SPARSE por pregunta')
plt.xlabel('Pregunta')
plt.ylabel('Diferencia de promedio')
plt.xticks(range(len(delta.index)), [f'P{i}' for i in delta.index])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'delta_rrf_sparse.png', dpi=180, bbox_inches='tight')
plt.show()

exported_files = {
    'latex_table': latex_output_path,
    'comparative_plot': OUTPUT_DIR / 'comparativa_general.png',
    'delta_plot': OUTPUT_DIR / 'delta_rrf_sparse.png',
}

exported_files

