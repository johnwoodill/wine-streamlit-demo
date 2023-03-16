import pandas as pd  
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score


def leave_one_out_cv(model, X, y):
    """
    Performs leave-one-out cross-validation on a given model using the provided input data.
    
    Parameters:
        - model: A scikit-learn model object with a `fit` and `predict` method.
        - X: A numpy array or pandas DataFrame containing the input features.
        - y: A numpy array or pandas Series containing the target variable.
    
    Returns:
        - scores: A numpy array of length `n` (where `n` is the number of samples in the input data)
                  containing the evaluation score for each fold of cross-validation.
    """
    from sklearn.model_selection import LeaveOneOut

    X = X.values
    y = y['Yield'].ravel()
    
    loo = LeaveOneOut()
    scores = []
    
    pred_ = []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        # score = model.score(X_test, y_test)
        pred = float(model.predict(X_test))
        pred_.append(pred)
    
    return pd.Series(pred_)




def get_features_labels(dat, cols):
    moddat = dat[cols]

    # Get lag effects for LeafN and PWMRow
    moddat = moddat.sort_values(['Year']).reset_index(drop=True)
    moddat['lag1_LeafN'] = moddat.groupby(['Vineyard', 'Treatment'])['LeafN'].shift(1)
    moddat['lag1_PWMRow'] = moddat.groupby(['Vineyard', 'Treatment'])['PWMRow'].shift(1)

    # Check lag
    lag_check = moddat[(moddat['Vineyard'] == 'JacobHart') & (moddat['Treatment'] == '1cls')].reset_index(drop=True)
    assert lag_check['lag1_LeafN'].shift(-1)[0] == lag_check['LeafN'][0]
    assert lag_check['lag1_LeafN'].shift(-1)[1] == lag_check['LeafN'][1]
    assert lag_check['lag1_LeafN'].shift(-1)[2] == lag_check['LeafN'][2]
    assert lag_check['lag1_PWMRow'].shift(-1)[0] == lag_check['PWMRow'][0]
    assert lag_check['lag1_PWMRow'].shift(-1)[1] == lag_check['PWMRow'][1]
    assert lag_check['lag1_PWMRow'].shift(-1)[2] == lag_check['PWMRow'][2]

    # Get dummies
    # vineyards = pd.get_dummies(moddat['Vineyard'])
    rootstocks = pd.get_dummies(moddat['Rootstock'], prefix='root')
    treatments = pd.get_dummies(moddat['Treatment'], prefix='treat')

    # Drop columns and attach dummies
    moddat = moddat.drop(columns=['Year', 'Vineyard', 'Rootstock', 'Treatment', 'PWMRow'])
    moddat = pd.concat([moddat, rootstocks, treatments], axis=1)
    moddat = moddat.dropna()

    X = moddat.drop(columns=['Yield'])
    y = moddat[['Yield']]

    # 1cls means 1 cluster was left out per shoot. So if the cluster initially had 3, then 2 of them are removed.

    return X, y


def gen_prediction(LeafN, LeafP, LeafK, Y_Tavg, Y_Pr, lag1_LeafN,
       lag1_PWMRow, root_101_14, root_3309, root_44_53, root_OWNR,
       root_RIPG, root_SHWM, treat_1cls, treat_2cls, treat_NoThin):

    # rfm = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=42)
    rfm = RandomForestRegressor(n_estimators=20, max_depth=8, random_state=42)

    cols = ['Vineyard', 'Treatment', 'Yield', 'Year', 'LeafN', 'LeafP', 'LeafK', 
            'Y_Tavg', 'Y_Pr', 'PWMRow', 'Rootstock']

    X, y = get_features_labels(dat, cols)

    nutr_model = rfm.fit(X.values, y['Yield'].ravel())

    X_test = [LeafN, LeafP, LeafK, Y_Tavg, Y_Pr, lag1_LeafN,
              lag1_PWMRow, root_101_14, root_3309, root_44_53, root_OWNR,
              root_RIPG, root_SHWM, treat_1cls, treat_2cls, treat_NoThin]

    X_test = [float(np.array(x)) for x in X_test]

    X_test = pd.Series(X_test).ravel().reshape(1, -1)

    y_pred = nutr_model.predict(X_test)
    return float(y_pred)




def get_r2():
    rfm = RandomForestRegressor(n_estimators=20, max_depth=8, random_state=42)
    # rfm = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=42)
    
    cols = ['Vineyard', 'Treatment', 'Yield', 'Year', 'LeafN', 'LeafP', 'LeafK', 
        'Y_Tavg', 'Y_Pr', 'PWMRow', 'Rootstock']

    X, y = get_features_labels(dat, cols)

    print("Cross-validating using Leave-One-Out Cross-validation")
    loo_pred = leave_one_out_cv(rfm, X, y)

    return r2_score(y, loo_pred)



# Load data
dat = pd.read_csv("data/Louis_All_variables_ML_ready_V3.csv")

# cols = ['Vineyard', 'Treatment', 'Yield', 'Year', 'LeafN', 'LeafP', 'LeafK', 
#         'Y_Tavg', 'Y_Pr', 'PWMRow', 'Rootstock']

# get_r2()



# 'Unnamed: 0', 'Year', 'Company', 'Vineyard', 'Treatment', 'Rep',
# 'TreatmentAge', 'ShootCt', 'ClusterCt', 'FFNS', 'ShootMCane',
# 'ShootFtCane', 'ClusterMCane', 'ClusterFtCane', 'ShootCt.1',
# 'PreClusterCt', 'PostClusterCt', 'PercThin', 'FFNS.1', 'ShootMRow',
# 'ShootFtRow', 'ShootMCane.1', 'ShootFtCane.1', 'PreClMRow',
# 'PreClFtRow', 'PreClMCane', 'PreClFtCane', 'PostClMRow', 'PostClFtRow',
# 'PostClMCane', 'PostClFtCane', 'LeafN', 'LeafP', 'LeafK', 'LeafCa',
# 'LeafMg', 'LeafB', 'LeafCu', 'LeafFe', 'LeafMn', 'LeafZn', 'LeafNa',
# 'PetN', 'PetP', 'PetK', 'PetCa', 'PetMg', 'PetB', 'PetCu', 'PetFe'
# 'PetZn', 'PetNa', 'TSS', 'pH', 'TA', 'ammoniaN', 'aaN', 'YAN',
# 'HarvestDate', 'ClusterCt.1', 'ClusterWt', 'Yield', 'KgMRow', 'LbFtRow',
# 'KgMCane', 'LbFtCane', 'ClusterMRow', 'ClusterFtRow', 'ClusterMCane.1',
# 'ClusterFtCane.1', 'ShootCt.2', 'PW', 'Canewt', 'PWMRow', 'PWFtRow',
# 'PWMCane', 'PWFtCane', 'ShootMRow.1', 'ShootFtRow.1', 'ShootMCane.2',
# 'ShootFtCane.2', 'Ravaz', 'anthos.std', 'tannins.std', 'phenolics.std',
# 'grid_num', 'ID', 'Y_Tavg', 'Y_Tmin', 'Y_Tmax', 'Y_Pr', 'GS_Tavg',
# 'GS_Tmin', 'GS_Tmax', 'GS_Pr', 'GS_Hug', 'GS_Gdd', 'DS_Pr', 'DS_Pr2mm'
# 'BB_Tmin', 'BB_Tmax', 'BB_Pr', 'BB_Tmin0C', 'FLO_Tavg', 'FLO_Tmin',
# 'FLO_Tmax', 'FLO_Pr', 'FLO_Pr2mm', 'FLO_Tmax35C', 'VER_Tavg',
# 'VER_Tmin', 'VER_Tmax', 'VER_Pr', 'VER_Tmax35C', 'H_Tavg', 'H_Tmin',
# 'H_Tmax', 'H_Pr', 'H_Pr2mm', 'H_Tmax35C', 'DEC_Y_Tavg', 'DEC_Y_Tmin',
# 'DEC_Y_Tmax', 'DEC_Y_Pr', 'DEC_GS_Tavg', 'DEC_GS_Tmin', 'DEC_GS_Tmax',
# 'DEC_GS_Pr', 'DEC_GS_Hug', 'DEC_GS_Gdd', 'DEC_DS_Pr', 'DEC_DS_Pr2mm',
# 'GS_Etp', 'BB_Etp', 'FLO_Etp', 'VER_Etp', 'H_Etp', 'DEC_GS_Etp',
# 'BB_B_Hu', 'B_V_Hu', 'V_H_Hu', 'BB_H_Hu', 'GS_XL_Hu', 'BB_B_nDays',
# 'B_V_nDays', 'V_H_nDays', 'BB_H_nDays', 'INI_Tavg'
# 'INI_Tmax', 'INI_Ra_net', 'INI_Pr', 'FLO_Ra_net', 'VER_Water_deficit',
# 'lon', 'lat'


# gen_prediction( 
#     LeafN=5, 
#     LeafP=0.3, 
#     LeafK=0.6, 
#     Y_Tavg=18, 
#     Y_Pr=0.1,
#     lag1_LeafN=1.9,
#     lag1_PWMRow=0.9,
#     root_101_14=0,
#     root_3309=0,
#     root_44_53=0,
#     root_OWNR=0,
#     root_RIPG=0,
#     root_SHWM=0,
#     treat_1cls=0,
#     treat_2cls=0,
#     treat_NoThin=0)


