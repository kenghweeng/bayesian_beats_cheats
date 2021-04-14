import pandas as pd
import numpy as np

def nodes1(df):
    '''
    Example usage:
    from src import preprocess

    df = pd.read_csv('data/unified_node_data.csv', keep_default_na=False)
    df = preprocess.nodes1(df)
    '''
    df = df.copy()
    # 1. What is your major, if outside of Science, use Faculty column. (1-hot encoding)
    others = ['Others (outside Science Faculty)', 'Others (in Science Faculty)', '', '-']
    df['What is your major?'] = df['What is your major?'].str.split(';').apply(set).apply(';'.join)
    df['major_or_faculty_if_unknown'] = df[['faculty', 'What is your major?']].apply(lambda r: r['faculty'] if r['What is your major?'] in others else r['What is your major?'], axis='columns')

    # 2. Year of Study (based on admit year, do smart extrapolation)
    batch2year = {2023: 20, 1935: 20, 1821: 19}
    df['year_of_study'] = df.apply(lambda r: max(1, 1 + batch2year[r['batch']] - r['admit_year']//100), axis='columns')

    # 3. Participation
    df['participation'] = df['participation'].replace('', np.nan).replace('-', np.nan).astype(float)

    # 4. PE_percent (special value NaN for missing values)
    # 5. Finals_percent  (special value NaN for missing values)
    # 6. midterms_percent  (special value NaN for missing values)
    df[['pe_percent', 'finals_percent', 'midterms_percent']] = df[['pe_percent', 'finals_percent', 'midterms_percent']].replace('', np.nan).replace('-', np.nan).astype(float)/100

    # 7. AFAST (binary)
    df['afast'] = df['afast'].replace('', np.nan).replace('-', np.nan).apply(lambda x: 0 if x == 'False' else 1 if x == 'True' else x).astype("Int64")

    # 8. Level_Min_Max
    # 9. EXP_Min_Max
    df[['level_min_max', 'exp_min_max']] = df[['level_min_max', 'exp_min_max']].replace('', np.nan).replace('-', np.nan).astype(float)

    # 10. Tutorial EXP 9 columns
    df[['t01_exp', 't02_exp', 't03_exp', 't04_exp', 't05_exp', 't06_exp', 't07_exp', 't08_exp', 't09_exp', 't10_exp']] = df[['t01_exp', 't02_exp', 't03_exp', 't04_exp', 't05_exp', 't06_exp', 't07_exp', 't08_exp', 't09_exp', 't10_exp']].replace('', np.nan).replace('-', np.nan).astype(float).astype("Int64")

    # 11. Num videos (Jon says take it with pinch of salt, varies based on whether sem was remote)
    # 12. Avg_videos_completed (Jon says take it with pinch of salt, varies based on whether sem was remote)
    df['num_videos'] = df['num_videos'].replace('', np.nan).replace('-', np.nan).astype(float).astype("Int64")
    df['avg_videos_completion'] = df['avg_videos_completion'].replace('', np.nan).replace('-', np.nan).astype(float)/100

    # 13. confessed assignments
    # 14. num_confessed_assignments
    df['confessed_assignments'] = df['confessed_assignments'].str.split(',')
    df['num_confessed_assignments'] = df['num_confessed_assignments'].astype(int)

    # 15. Batch (one-hot encode)
    df['batch'] = df['batch'].astype(int)

    tgt_cols = ['name', 'batch', 'major_or_faculty_if_unknown', 'year_of_study', 'participation', 'pe_percent', 'finals_percent', 'midterms_percent', 'afast', 'level_min_max', 'exp_min_max', 't01_exp', 't02_exp', 't03_exp', 't04_exp', 't05_exp', 't06_exp', 't07_exp', 't08_exp', 't09_exp', 't10_exp', 'num_videos', 'avg_videos_completion', 'confessed_assignments', 'num_confessed_assignments']
    df = df[tgt_cols]

    return pd.get_dummies(df, columns=['batch', 'major_or_faculty_if_unknown'], prefix=['batch', 'major'], prefix_sep='_')

def nodes_filter(node_df, edge_list_df):
    '''
    Example usage:
    from src import preprocess

    node_df = pd.read_csv('data/unified_node_data.csv', keep_default_na=False)
    edge_df = pd.read_csv('data/max_edge_weights.csv')
    df = preprocess.nodes_filter(node_df, edge_df)
    '''
    edge_names = set(edge_list_df['NodeID1']) | set(edge_list_df['NodeID2'])
    mask = node_df['name'].isin(edge_names)
    return node_df[mask]