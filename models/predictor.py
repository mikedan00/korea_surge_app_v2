# models/predictor.py
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, VotingClassifier,
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_absolute_error
from core.features import FEATURE_COLS
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

RS = 42

def build_classifier(rs=RS):
    est = [
        ("rf", RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=3,
             class_weight="balanced_subsample",random_state=rs,n_jobs=-1)),
        ("et", ExtraTreesClassifier(n_estimators=250,max_depth=8,min_samples_leaf=3,
             class_weight="balanced",random_state=rs,n_jobs=-1)),
        ("gb", GradientBoostingClassifier(n_estimators=150,max_depth=5,
             learning_rate=0.05,subsample=0.8,random_state=rs)),
    ]
    if HAS_XGB:
        est.append(("xgb", XGBClassifier(n_estimators=150,max_depth=6,learning_rate=0.05,
            subsample=0.8,colsample_bytree=0.8,scale_pos_weight=10,
            use_label_encoder=False,eval_metric="logloss",
            random_state=rs,n_jobs=-1,verbosity=0)))
    if HAS_LGB:
        est.append(("lgb", LGBMClassifier(n_estimators=150,max_depth=6,learning_rate=0.05,
            subsample=0.8,class_weight="balanced",random_state=rs,n_jobs=-1,verbose=-1)))
    return CalibratedClassifierCV(
        VotingClassifier(estimators=est,voting="soft",n_jobs=-1),
        method="isotonic",cv=3)

def build_regressor(rs=RS):
    from sklearn.ensemble import VotingRegressor
    est = [
        ("rf", RandomForestRegressor(n_estimators=300,max_depth=8,min_samples_leaf=3,random_state=rs,n_jobs=-1)),
        ("et", ExtraTreesRegressor(n_estimators=250,max_depth=8,min_samples_leaf=3,random_state=rs,n_jobs=-1)),
        ("gb", GradientBoostingRegressor(n_estimators=150,max_depth=5,learning_rate=0.05,subsample=0.8,random_state=rs)),
    ]
    if HAS_XGB:
        est.append(("xgb", XGBRegressor(n_estimators=150,max_depth=6,learning_rate=0.05,
            subsample=0.8,colsample_bytree=0.8,random_state=rs,n_jobs=-1,verbosity=0)))
    if HAS_LGB:
        est.append(("lgb", LGBMRegressor(n_estimators=150,max_depth=6,learning_rate=0.05,
            subsample=0.8,random_state=rs,n_jobs=-1,verbose=-1)))
    return VotingRegressor(estimators=est,n_jobs=-1)

def ts_cv_score(X, y, mode="clf", n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr,val in tscv.split(X):
        Xtr,Xval = X.iloc[tr],X.iloc[val]
        ytr,yval = y.iloc[tr],y.iloc[val]
        if mode=="clf":
            if len(set(ytr))<2 or ytr.sum()<2: continue
            try:
                m=build_classifier(); m.fit(Xtr,ytr)
                prob=m.predict_proba(Xval)[:,1]
                if len(set(yval))==2: scores.append(roc_auc_score(yval,prob))
            except: continue
        else:
            try:
                m=build_regressor(); m.fit(Xtr,ytr)
                mae=mean_absolute_error(yval,m.predict(Xval))
                scores.append(1-min(mae*10,1))
            except: continue
    return float(np.mean(scores)) if scores else np.nan

def predict_ticker(feat_df: pd.DataFrame, target_pct: float = 0.15) -> dict:
    X = feat_df[FEATURE_COLS]
    y_cls = feat_df["target_15up"]
    y_reg = feat_df["next_ret"] * 100

    X_tr  = X.iloc[:-1]; Xp = X.iloc[[-1]]
    yc_tr = y_cls.iloc[:-1]; yr_tr = y_reg.iloc[:-1]

    pos_count = int(yc_tr.sum()); base_rate = float(yc_tr.mean())
    parts = ["RF","ET","GB"] + (["XGB"] if HAS_XGB else []) + (["LGB"] if HAS_LGB else [])
    model_info = "+".join(parts) + " (Calibrated Ensemble)"

    if pos_count < 3 or len(set(yc_tr)) < 2:
        prob_up15=base_rate; cv_auc=np.nan
    else:
        try:
            clf=build_classifier(); clf.fit(X_tr,yc_tr)
            prob_up15=float(clf.predict_proba(Xp)[0][1])
            cv_auc=ts_cv_score(X_tr,yc_tr,mode="clf") if pos_count>=5 else np.nan
        except:
            prob_up15=base_rate; cv_auc=np.nan

    try:
        reg=build_regressor(); reg.fit(X_tr,yr_tr)
        pred_ret_pct=float(reg.predict(Xp)[0])
        cv_reg=ts_cv_score(X_tr,yr_tr,mode="reg") if len(X_tr)>=150 else np.nan
        indiv=[float(m.predict(Xp)[0]) for _,m in reg.estimators_]
        conf_band=float(np.std(indiv)) if indiv else abs(pred_ret_pct*0.3)
    except:
        pred_ret_pct=float(yr_tr.tail(20).mean())
        conf_band=float(yr_tr.tail(20).std()); cv_reg=np.nan

    up_hist=yr_tr[yr_tr>0].mean() if (yr_tr>0).sum()>0 else np.nan
    dn_hist=abs(yr_tr[yr_tr<0].mean()) if (yr_tr<0).sum()>0 else np.nan

    return {
        "prob_up15":   round(prob_up15,4),
        "pred_ret_pct":round(pred_ret_pct,2),
        "pred_up_pct": round(max(0.0,pred_ret_pct),2),
        "pred_dn_pct": round(max(0.0,-pred_ret_pct),2),
        "conf_band":   round(conf_band,2),
        "cv_auc":      round(cv_auc,3) if not np.isnan(cv_auc) else None,
        "cv_reg":      round(cv_reg,3) if not np.isnan(cv_reg) else None,
        "base_rate":   round(base_rate,4),
        "pos_count":   pos_count,
        "up_hist_avg": round(up_hist,2) if not np.isnan(up_hist) else None,
        "dn_hist_avg": round(dn_hist,2) if not np.isnan(dn_hist) else None,
        "model_info":  model_info,
    }
