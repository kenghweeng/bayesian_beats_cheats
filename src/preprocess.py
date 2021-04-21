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
    missing_samples = np.arange(n_samples)
    missing_features = rng.choice(n_features, n_samples, replace=True)
    X_missing = X_scaled.copy()
    X_missing[missing_samples, missing_features] = np.nan

    X_to_check = ~np.isnan(X_scaled) & np.isnan(X_missing)

    estimators = [
        LinearRegression(),
        BayesianRidge(),
        KNeighborsRegressor(),
    ]
    data = []
    for i in range(10):
        iter_seed = rng.randint(0, 10**6)
        random_estimators = [
            DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=iter_seed),
            ExtraTreesRegressor(n_estimators=20, max_depth=5, random_state=seed), # takes ~1min to run at 100 trees
            RandomForestRegressor(n_estimators=20, max_depth=5, random_state=seed), # takes ~1min to run at 100 trees
        ]

        row = []
        for est in estimators: # these are deterministic, no need to recompute
            if i == 0:
                imp_mean = IterativeImputer(estimator=est, max_iter=100, random_state=iter_seed)
                X_pred = np.clip(imp_mean.fit_transform(X_missing), 0, 1)
                est.error_value = mean_squared_error(X[X_to_check], scaler.inverse_transform(X_pred)[X_to_check], squared=False)
            row.append(est.error_value)
            print(est.__class__.__name__)
        for est in random_estimators:
            imp_mean = IterativeImputer(estimator=est, max_iter=100, random_state=iter_seed)
            X_pred = np.clip(imp_mean.fit_transform(X_missing), 0, 1)
            row.append(mean_squared_error(X_scaled[X_to_check], scaler.inverse_transform(X_pred)[X_to_check], squared=False))
        data.append(row)
    df_stats = pd.DataFrame(data, columns=[est.__class__.__name__ for est in estimators+random_estimators]).T
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

    df_node = pd.read_csv('data/unified_node_data.csv', keep_default_na=False)
    df_edge = pd.read_csv('data/max_edge_weights.csv')
    df_formatted = preprocess.nodes1(df_node)
    df_clean = preprocess.nodes_filter(df_formatted, df_edge)
    df_impute = preprocess.impute(df_clean)

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