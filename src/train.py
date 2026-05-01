import logging
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

from . import data_loader
from . import feature_engineering as fe
from .models import get_lgbm_model, get_xgb_model, get_cat_model
from .utils import setup_logging, plot_test_forecast, generate_shap_plots

logger = logging.getLogger(__name__)


def run_pipeline(
    data_dir: str,
    output_dir: str = "models",
    log_dir: str = "logs",
    n_trials: int = 0,     # Ignored in this version
    n_folds: int = 5,
    val_months: int = 0,   # Ignored, using TimeSeriesSplit
    seed: int = 42,
    skip_tuning: bool = True,
) -> pd.DataFrame:
    setup_logging(log_dir)
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("STEP 1: Loading raw data")
    logger.info("=" * 70)
    data = data_loader.load_all(data_dir)

    sales = data["sales"].copy()
    sales["Date"] = pd.to_datetime(sales["Date"])
    sales = sales.sort_values("Date").reset_index(drop=True)
    
    submission = data["sample_submission"].copy()
    submission["Date"] = pd.to_datetime(submission["Date"])

    logger.info("=" * 70)
    logger.info("STEP 2: Building Features (Occam's Razor 10 Features)")
    logger.info("=" * 70)
    
    train_df = fe.build_features(sales)
    test_df = fe.build_features(submission)
    feature_cols = fe.get_feature_columns()
    
    logger.info("=" * 70)
    logger.info("STEP 3: Training Ensemble (LightGBM, XGBoost, CatBoost) via TimeSeriesSplit")
    logger.info("=" * 70)

    tscv = TimeSeriesSplit(n_splits=n_folds)
    
    def train_ensemble_for_target(target_name):
        logger.info(f"Training ensemble for {target_name}...")
        models_lgb = []
        models_xgb = []
        models_cat = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(train_df)):
            X_tr = train_df.loc[train_idx, feature_cols]
            X_va = train_df.loc[val_idx, feature_cols]

            y_tr = np.log1p(train_df.loc[train_idx, target_name])
            y_va_log = np.log1p(train_df.loc[val_idx, target_name])

            # 1. LightGBM
            m_lgb = get_lgbm_model()
            m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va_log)], callbacks=[lgb.early_stopping(100, verbose=False)])
            models_lgb.append(m_lgb)

            # 2. XGBoost
            m_xgb = get_xgb_model()
            m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va_log)], verbose=False)
            models_xgb.append(m_xgb)

            # 3. CatBoost
            m_cat = get_cat_model()
            m_cat.fit(X_tr, y_tr, eval_set=[(X_va, y_va_log)], early_stopping_rounds=100, verbose=False)
            models_cat.append(m_cat)
            
            logger.info(f"  Fold {fold+1}/{n_folds} completed.")
            
        return {'lgb': models_lgb, 'xgb': models_xgb, 'cat': models_cat}

    models_rev = train_ensemble_for_target("Revenue")
    models_cogs = train_ensemble_for_target("COGS")

    logger.info("=" * 70)
    logger.info("STEP 4: Direct Forecasting & Simple Averaging Ensemble")
    logger.info("=" * 70)

    def predict_ensemble(models_dict):
        all_preds = []
        for m in models_dict['lgb']: all_preds.append(np.expm1(m.predict(test_df[feature_cols])))
        for m in models_dict['xgb']: all_preds.append(np.expm1(m.predict(test_df[feature_cols])))
        for m in models_dict['cat']: all_preds.append(np.expm1(m.predict(test_df[feature_cols])))
        return np.mean(all_preds, axis=0)

    test_preds_rev = predict_ensemble(models_rev)
    test_preds_cogs = predict_ensemble(models_cogs)

    # Ràng buộc tài chính (Business constraints)
    test_preds_rev = np.clip(test_preds_rev, 0, None)
    test_preds_cogs = np.clip(test_preds_cogs, 0, None)
    test_preds_cogs = np.minimum(test_preds_cogs, test_preds_rev)

    submission["Revenue"] = test_preds_rev.round(2)
    submission["COGS"] = test_preds_cogs.round(2)

    sub_path = os.path.join(output_dir, "submission.csv")
    submission.to_csv(sub_path, index=False)
    logger.info("Submission saved: %s (%d rows)", sub_path, len(submission))
    
    logger.info("Submission Revenue stats:")
    logger.info(f"  Mean: {submission['Revenue'].mean():,.0f} | Std: {submission['Revenue'].std():,.0f}")

    plot_test_forecast(
        train_df["Date"], train_df["Revenue"].values,
        submission["Date"], submission["Revenue"].values,
        title="Revenue Forecast (67-Point Ensemble)",
        save_path=os.path.join(output_dir, "forecast_revenue.png"),
    )
    
    plot_test_forecast(
        train_df["Date"], train_df["COGS"].values,
        submission["Date"], submission["COGS"].values,
        title="COGS Forecast (67-Point Ensemble)",
        save_path=os.path.join(output_dir, "forecast_cogs.png"),
    )

    # Generate SHAP plots for the best model (last fold LGBM)
    logger.info("Generating SHAP feature importance plots...")
    generate_shap_plots(
        models_rev['lgb'][-1], 
        train_df[feature_cols], 
        save_dir=output_dir, 
        model_name="revenue_drivers"
    )
    
    generate_shap_plots(
        models_cogs['lgb'][-1], 
        train_df[feature_cols], 
        save_dir=output_dir, 
        model_name="cogs_drivers"
    )

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info("Pipeline complete! Total time: %.1f minutes", elapsed / 60)
    logger.info("=" * 70)

    return submission
