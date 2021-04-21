import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge

def nodes1(df):
    '''
    Example usage:
    import pandas as pd
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
    import pandas as pd
    from src import preprocess

    node_df = pd.read_csv('data/unified_node_data.csv', keep_default_na=False)
    edge_df = pd.read_csv('data/max_edge_weights.csv')
    df = preprocess.nodes_filter(node_df, edge_df)
    '''
    edge_names = set(edge_list_df['NodeID1']) | set(edge_list_df['NodeID2'])
    mask = node_df['name'].isin(edge_names)
    return node_df[mask]

def impute_stats(node_df):
    '''
    Motivation: Check performance of different imputers.
             1) normalize column-wise
             2) randomly remove some known values
             3) impute
             4) compute the RMSE of the imputed values from the true values
             5) rank estimators according to average RMSE
    Example usage:
    import pandas as pd
    from src import preprocess

    df_node = pd.read_csv('data/unified_node_data.csv', keep_default_na=False)
    df_edge = pd.read_csv('data/max_edge_weights.csv')
    df_formatted = preprocess.nodes1(df_node)
    df = preprocess.nodes_filter(df_formatted, df_edge)
    preprocess.impute_stats(df)
    '''
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    from sklearn.linear_model import LinearRegression, BayesianRidge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import MinMaxScaler

    seed = 0

    node_df = node_df.copy()
    X = node_df.select_dtypes(include='number').astype(float).values

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    n_samples, n_features = X_scaled.shape

    rng = np.random.RandomState(seed)

    data = []
    for i in range(5):
        iter_seed = rng.randint(0, 10**6)

        missing_samples = np.arange(n_samples)
        missing_features = rng.choice(n_features, n_samples, replace=True)
        X_missing = X_scaled.copy()
        X_missing[missing_samples, missing_features] = np.nan

        X_to_check = ~np.isnan(X_scaled) & np.isnan(X_missing)

        estimators = [
            KNeighborsRegressor(n_neighbors=3, weights='distance', n_jobs=-1),  # 0
            KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1),  # 1
            KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=-1), # 2
            KNeighborsRegressor(n_neighbors=20, weights='distance', n_jobs=-1), # 3
            KNeighborsRegressor(n_neighbors=3, weights='uniform', n_jobs=-1),   # 4
            KNeighborsRegressor(n_neighbors=5, weights='uniform', n_jobs=-1),   # 5
            KNeighborsRegressor(n_neighbors=10, weights='uniform', n_jobs=-1),  # 6
            KNeighborsRegressor(n_neighbors=20, weights='uniform', n_jobs=-1),  # 7
            # LinearRegression(),
            # BayesianRidge(),
            # DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=iter_seed),
            # ExtraTreesRegressor(n_estimators=20, max_depth=5, random_state=seed), # takes ~1min to run at 100 trees
            # RandomForestRegressor(n_estimators=20, max_depth=5, random_state=seed), # takes ~1min to run at 100 trees
        ]

        row = []
        for est in estimators:
            imp_mean = IterativeImputer(estimator=est, max_iter=100, random_state=iter_seed)
            X_pred = np.clip(imp_mean.fit_transform(X_missing), 0, 1)
            rmse_err = mean_squared_error(X_scaled[X_to_check], scaler.inverse_transform(X_pred)[X_to_check], squared=False)
            row.append(rmse_err)
        data.append(row)
    df_stats = pd.DataFrame(data, columns=[f'{est.__class__.__name__}{i:02}' for i, est in enumerate(estimators)]).T
    df_stats['avg_rmse'] = df_stats.mean(axis='columns')
    return df_stats.sort_values('avg_rmse')

def impute(node_df, estimator=BayesianRidge()):
    '''
    Example usage:
    import pandas as pd
    from src import preprocess

    df_node = pd.read_csv('data/unified_node_data.csv', keep_default_na=False)
    df_edge = pd.read_csv('data/max_edge_weights.csv')
    df_formatted = preprocess.nodes1(df_node)
    df_clean = preprocess.nodes_filter(df_formatted, df_edge)
    df_impute = preprocess.impute(df_clean)
    '''
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import LinearRegression, BayesianRidge
    from sklearn.preprocessing import MinMaxScaler

    seed = 0
    dtypes = node_df.dtypes
    node_df = node_df.copy()
    numeric_cols = node_df.select_dtypes(include='number').columns

    X = node_df[numeric_cols].astype(float).values

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    imp_mean = IterativeImputer(estimator=estimator, max_iter=100, random_state=0)
    X_pred = scaler.inverse_transform(np.clip(imp_mean.fit_transform(X_scaled), 0, 1))
    node_df[numeric_cols] = X_pred

    # recast types
    for col, dtype in zip(node_df.columns, dtypes):
        if dtype == np.int64 or isinstance(dtype, pd.Int64Dtype):
            node_df[col] = node_df[col].round(decimals=0).astype(np.int64)
        else:
            node_df[col] = node_df[col].astype(dtype)
    return node_df

def stratified_train_val_test_split(node_df, val_size=0.2, test_size=0.1):
    '''
    Example usage:
    import pandas as pd
    from src import preprocess

    df_node = pd.read_csv('data/imputed_unified_node_data.csv', keep_default_na=False)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess.stratified_train_val_test_split(df_impute)
    '''
    from sklearn.model_selection import train_test_split
    seed = 0
    target_col = 'num_confessed_assignments'

    X           = node_df.drop(target_col, axis='columns')
    y           = node_df[target_col]
    stratify_on = node_df[target_col] > 0

    X_train, X_valtest, y_train, y_valtest = train_test_split(X,         y,         test_size=(val_size + test_size),           random_state=seed, stratify=stratify_on)

    stratify_on = y_valtest > 0
    X_val,   X_test,    y_val,   y_test    = train_test_split(X_valtest, y_valtest, test_size=test_size/(val_size + test_size), random_state=seed, stratify=stratify_on)
    return X_train, X_val, X_test, y_train, y_val, y_test

def downsample(df, tgt, random_state=0):
    groups = df.groupby(tgt)
    min_size = groups.size().min()
    return groups.sample(n=min_size, random_state=random_state)


def get_rank_weight(df):
    groupby_mission = df.groupby('mission')
    df['max_rank'] = groupby_mission['rank'].transform('count')
    df['weight_rank'] = np.exp(-(df['rank']/df['max_rank']))
    return df

def get_line_weight(df):
    df['weight_lines'] = 1 - np.exp(-df['lines']/10)
    return df

def get_percent_weights(df):
    df['weight_percent_max'] = df[['percent1', 'percent2']].max(axis='columns')/100
    df['weight_percent_min'] = df[['percent1', 'percent2']].min(axis='columns')/100
    df['weight_percent_mean'] = df[['percent1', 'percent2']].mean(axis='columns')/100
    groupby_mission = df.groupby('mission')
    max_mean  = groupby_mission['weight_percent_max'].mean()
    max_std   = groupby_mission['weight_percent_max'].std()
    min_mean  = groupby_mission['weight_percent_min'].mean()
    min_std   = groupby_mission['weight_percent_min'].std()
    mean_mean = groupby_mission['weight_percent_mean'].mean()
    mean_std  = groupby_mission['weight_percent_mean'].std()
    df['weight_percent_max_norm']  = df.apply(lambda r: (r['weight_percent_max'] - max_mean[r['mission']]) / max_std[r['mission']], axis="columns")
    df['weight_percent_min_norm']  = df.apply(lambda r: (r['weight_percent_min'] - min_mean[r['mission']]) / min_std[r['mission']], axis="columns")
    df['weight_percent_mean_norm'] = df.apply(lambda r: (r['weight_percent_mean'] - mean_mean[r['mission']]) / mean_std[r['mission']], axis="columns")
    max_mission_weights  = (1 - max_mean).to_dict()
    min_mission_weights  = (1 - min_mean).to_dict()
    mean_mission_weights = (1 - mean_mean).to_dict()
    df['weight_percent_max_weighted_mean']  = df.apply(lambda r: r['weight_percent_max'] * max_mission_weights[r['mission']], axis="columns")
    df['weight_percent_min_weighted_mean']  = df.apply(lambda r: r['weight_percent_min'] * min_mission_weights[r['mission']], axis="columns")
    df['weight_percent_mean_weighted_mean'] = df.apply(lambda r: r['weight_percent_mean'] * mean_mission_weights[r['mission']], axis="columns")
    df['weight_percent_max_norm_weighted_mean']  = df.apply(lambda r: r['weight_percent_max_norm'] * max_mission_weights[r['mission']], axis="columns")
    df['weight_percent_min_norm_weighted_mean']  = df.apply(lambda r: r['weight_percent_min_norm'] * min_mission_weights[r['mission']], axis="columns")
    df['weight_percent_mean_norm_weighted_mean'] = df.apply(lambda r: r['weight_percent_mean_norm'] * mean_mission_weights[r['mission']], axis="columns")

    
    max_mission_weights  = max_std.to_dict()
    min_mission_weights  = min_std.to_dict()
    mean_mission_weights = mean_std.to_dict()
    df['weight_percent_max_weighted_std']  = df.apply(lambda r: r['weight_percent_max'] * max_mission_weights[r['mission']], axis="columns")
    df['weight_percent_min_weighted_std']  = df.apply(lambda r: r['weight_percent_min'] * min_mission_weights[r['mission']], axis="columns")
    df['weight_percent_mean_weighted_std'] = df.apply(lambda r: r['weight_percent_mean'] * mean_mission_weights[r['mission']], axis="columns")
    df['weight_percent_max_norm_weighted_std']  = df.apply(lambda r: r['weight_percent_max_norm'] * max_mission_weights[r['mission']], axis="columns")
    df['weight_percent_min_norm_weighted_std']  = df.apply(lambda r: r['weight_percent_min_norm'] * min_mission_weights[r['mission']], axis="columns")
    df['weight_percent_mean_norm_weighted_std'] = df.apply(lambda r: r['weight_percent_mean_norm'] * mean_mission_weights[r['mission']], axis="columns")
    return df

def get_percent_weights(df):
    df['weight_percent_max'] = df[['percent1', 'percent2']].max(axis='columns')/100
    df['weight_percent_min'] = df[['percent1', 'percent2']].min(axis='columns')/100
    df['weight_percent_mean'] = df[['percent1', 'percent2']].mean(axis='columns')/100

    groupby_mission = df.groupby('mission')
    for agg in ['max', 'min', 'mean']:
        df[f'{agg}_mean']  = groupby_mission[f'weight_percent_{agg}'].transform('mean')
        df[f'{agg}_std']   = groupby_mission[f'weight_percent_{agg}'].transform('std')
        df[f'weight_percent_{agg}_norm']  = (df[f'weight_percent_{agg}'] - df[f'{agg}_mean']) / df[f'{agg}_std']

        df[f'weight_percent_{agg}_weighted_mean']       = df[f'weight_percent_{agg}'] * (1 - df[f'{agg}_mean'])
        df[f'weight_percent_{agg}_norm_weighted_mean']  = df[f'weight_percent_{agg}_norm'] * (1 - df[f'{agg}_mean'])

        df[f'weight_percent_{agg}_weighted_std']        = df[f'weight_percent_{agg}'] *df[f'{agg}_std']
        df[f'weight_percent_{agg}_norm_weighted_std']   = df[f'weight_percent_{agg}_norm'] * df[f'{agg}_std']
    return df

def get_combined_weights(df):
    for flavour in ['', '_norm']:
        for agg in ['max', 'min', 'mean']:
            base = f'weight_percent_{agg}{flavour}'
            df[f'{base}_rank']       = df[f'{base}'] * df['weight_rank']
            df[f'{base}_lines']      = df[f'{base}'] * df['weight_lines']
            df[f'{base}_rank_lines'] = df[f'{base}_rank'] * df['weight_lines']
            for weight_type in ['mean', 'std']:
                df[f'{base}_weighted_{weight_type}_rank']       = df[f'{base}_weighted_{weight_type}'] * df['weight_rank']
                df[f'{base}_weighted_{weight_type}_lines']      = df[f'{base}_weighted_{weight_type}'] * df['weight_lines']
                df[f'{base}_weighted_{weight_type}_rank_lines'] = df[f'{base}_weighted_{weight_type}_rank'] * df['weight_lines']
    return df

def get_node_features_from_moss(df, moss_files):
    '''
    Example usage:
    pairwise_moss = ['raw_data/1821_1935.csv', 'raw_data/1821_2023.csv', 'raw_data/1935_2023.csv']
    df_node = pd.read_csv('data/imputed_unified_node_data.csv', keep_default_na=False)

    df_new_node = preprocess.get_node_features_from_moss(df_node, pairwise_moss)
    '''
    from scipy.sparse.csgraph import csgraph_from_dense, shortest_path

    df_moss = []
    for fn in moss_files:
        _df_moss = pd.read_csv(fn)
        mask = _df_moss['name1'] != _df_moss['name2']
        _df_moss = _df_moss[mask]
        _df_moss['rank'] = _df_moss['url'].str.rsplit('match', 1).str[-1].str[:-5].astype(int)
        df_moss.append(get_combined_weights(get_percent_weights(get_line_weight(get_rank_weight(_df_moss)))))
    df_moss = pd.concat(df_moss)
    df_moss.drop_duplicates(subset=['mission', 'name1', 'name2'], keep="first", inplace=True)

    name_subset = set(df['name'])

    weight_type = 'weight_percent_max_norm_weighted_std_rank_lines'
    mask = (df_moss['name1'].isin(name_subset) & df_moss['name2'].isin(name_subset) & (df_moss[weight_type] > 0))
    edge_list = df_moss[mask][['name1', 'name2', 'mission', weight_type]].values

    missions2idx = {mission: i for i, mission in enumerate(sorted((set(df_moss['mission']))))}
    names2idx = {name: i for i, name in enumerate(df['name'])}

    shape = (len(missions2idx), len(names2idx), len(names2idx))

    A = np.zeros(shape=shape, dtype=float)
    for name1, name2, mission, weight in edge_list:
        m = missions2idx[mission]
        i = names2idx[name1]
        j = names2idx[name2]
        A[m, i, j] += np.log(weight)
    A[A==0] = np.inf
    A = np.exp(A)
    
    for i in range(len(missions2idx)):
        G2_sparse = csgraph_from_dense(A[i], null_value=np.inf)
        sg = shortest_path(G2_sparse, directed=False)
        sg[sg == np.inf] = 1
        A[i] = 1 - sg
    new_features = pd.DataFrame(np.hstack(A), index=df['name'])
    # new_features = new_features.merge(df[['name', 'num_confessed_assignments']], left_index=True, right_on='name')
    return new_features.reset_index()