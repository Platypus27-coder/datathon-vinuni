import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

def get_lgbm_model():
    params = dict(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    return lgb.LGBMRegressor(**params)

def get_xgb_model():
    params = dict(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        tree_method='hist'
    )
    return xgb.XGBRegressor(**params)

def get_cat_model():
    params = dict(
        iterations=1500,
        learning_rate=0.03,
        depth=7,
        random_seed=42,
        verbose=False
    )
    return CatBoostRegressor(**params)
