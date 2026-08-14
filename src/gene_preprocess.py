import pandas as pd
from io import StringIO

def extract_expression_only(filepath="C:\\Users\\lenovo\\OneDrive\\Documents\\coding\\BPD\\data\\raw\\GSE32472_series_matrix.txt"):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    start_idx = end_idx = None
    for i, line in enumerate(lines):
        if line.startswith('!series_matrix_table_begin'):
            start_idx = i + 1
        elif line.startswith('!series_matrix_table_end'):
            end_idx = i
            break

    expr_df = pd.read_csv(
        StringIO(''.join(lines[start_idx:end_idx])),
        sep='\t',
        index_col=0
    )
    expr_df.columns = expr_df.columns.str.strip('"')
    expr_df.index.name = 'probe_id'
    return expr_df

expr_df = extract_expression_only()
expr_df.to_csv('GSE32472_expression.csv')

print(expr_df.shape)
print(expr_df.head())