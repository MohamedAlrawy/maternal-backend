#!/usr/bin/env python3
# patients/management/commands/train_pph.py
# \"\"\"Django management command to train PPH predictor.

# Saves artifacts to <BASE_DIR>/artifacts:
#  - pph_pipeline.joblib
#  - pph_metrics.json
#  - pph_feature_importances.csv
#  - pph_rule_summary.json
# \"\"\"

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging

import joblib
import numpy as np
import pandas as pd

from django.core.management.base import BaseCommand
from django.conf import settings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix
)

try:
    from patients.models import Patient
except Exception:  # pragma: no cover
    Patient = None

logger = logging.getLogger(__name__)


# ---------- Helpers (parsing) ----------
def safe_parse_list(cell):
    if cell is None:
        return []
    if isinstance(cell, (list, tuple)):
        return [str(x).strip() for x in cell if str(x).strip()!='']
    if isinstance(cell, dict):
        return [str(v).strip() for v in cell.values() if str(v).strip()!='']
    if isinstance(cell, str):
        try:
            parsed = json.loads(cell)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()!='']
        except Exception:
            pass
        parts = re.split(r'[;,|]\s*', cell)
        return [p.strip() for p in parts if p.strip()!='']
    return []

def contains_any(lst, keys):
    if not lst:
        return False
    low = [str(s).lower() for s in lst]
    for k in keys:
        kl = k.lower()
        if any(kl in s for s in low):
            return True
    return False

def infer_total_cs_from_obstetric(obst_list):
    # \"\"\"Look for patterns in obstetric_history list to infer CS count.\"\"\"
    if not obst_list:
        return 0
    text = ' '.join(obst_list).lower()
    # common tokens
    if 'three' in text or '≥3' in text or '>=3' in text or '3+' in text or 'more than 2' in text:
        return 3
    if re.search(r'2\s*(c[\.\-/ ]?section|cs|c/s)', text) or 'two' in text or 'second cs' in text:
        return 2
    if re.search(r'1\s*(c[\.\-/ ]?section|cs|c/s)|previous c', text) or 'one' in text:
        return 1
    # fallback: count explicit numbers followed by cs/c-section
    m = re.findall(r'(\d+)\s*(?:c[\.\-/ ]?section|cs|c/s)', text)
    if m:
        try:
            return int(m[-1])
        except Exception:
            return 0
    return 0

# ---------- Deterministic PPH rules (weights & reasons) ----------
PPH_RULES = [
    # (predicate, weight_percent, reason)
    (lambda r: contains_any(r.get('obstetric_history', []), ['postpartum hemorrhage','pph']), 20, "History of prior PPH (15-25%)"),
    (lambda r: (r.get('parity') is not None and r.get('parity') >= 5) or contains_any(r.get('social', []), ['grand multipara']), 12, "Grand multipara (>=5) (10-15%)"),
    (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['multiple gestation','twin','twins','triplet']), 12, "Multiple gestation (10-15%)"),
    (lambda r: contains_any(r.get('liquor_flags', []), ['polyhydramnios','polihydramnios','polihydraminos']), 9, "Polyhydramnios (8-10%)"),
    (lambda r: (not pd.isna(r.get('estimated_fetal_weight_by_gm')) and float(r.get('estimated_fetal_weight_by_gm',0)) >= 4000), 10, "Estimated fetal weight >=4000g (8-12%)"),
    (lambda r: contains_any(r.get('obstetric_history', []), ['placenta abruption','abruption']), 12, "History of placental abruption (10-15%)"),
    (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['placenta previa','placenta praevia']), 15, "Placenta previa (10-20%)"),
    (lambda r: (not pd.isna(r.get('hb_g_dl')) and float(r.get('hb_g_dl',999)) < 7), 5, "Severe anemia (Hb<7) - indirect impact"),
    (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['pre-eclampsia','preeclampsia','pre eclampsia']), 7, "Pre-eclampsia (5-10%)"),
    (lambda r: (r.get('total_number_of_cs',0) > 1) or contains_any(r.get('obstetric_history', []), ['multiple c-section','2 c-section','two c-section']), 7, "Multiple prior CS (>1) (5-10%)"),
    (lambda r: contains_any(r.get('obstetric_history', []), ['obstructed','prolonged labor','prolonged']), 7, "Obstructed/prolonged labor (5-10%)"),
]

def pph_deterministic_check(rowdict):
    matches = []
    total = 0
    reasons = []
    for cond, weight, reason in PPH_RULES:
        try:
            if cond(rowdict):
                matches.append((weight, reason))
                total += weight
                reasons.append(reason)
        except Exception:
            continue
    return (len(matches)>0), total, reasons

# ---------- Data loading & feature engineering ----------
def load_patients_df(qs=None):
    if qs is None:
        if Patient is None:
            raise RuntimeError('Patient model not importable. Run inside Django.')
        qs = Patient.objects.all()
    rows = []
    simple_fields = [
        'id','age','parity','bmi','height','weight','total_number_of_cs',
        'mode_of_delivery','type_of_labor','perineum_integrity','instrumental_delivery','type_of_cs',
        'labor_duration_hours','placenta_location','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l',
        'fetus_number'
    ]
    list_fields = ['menternal_medical','obstetric_history','current_pregnancy_menternal','social','liquor']
    for p in qs:
        d = {}
        for f in simple_fields:
            d[f] = getattr(p, f, None)
        for lf in list_fields:
            d[lf] = getattr(p, lf, None)
        rows.append(d)
    df = pd.DataFrame(rows)
    return df

def normalize_lists(df):
    list_cols = ['menternal_medical','obstetric_history','current_pregnancy_menternal','social','liquor']
    for c in list_cols:
        if c not in df.columns:
            df[c] = [[] for _ in range(len(df))]
        else:
            df[c] = df[c].apply(safe_parse_list)
    df['liquor_flags'] = df['liquor'].apply(lambda L: [s.lower() for s in L if isinstance(s,str)])
    return df

def derive_flags(df):
    df['history_pph'] = df['obstetric_history'].apply(lambda L: contains_any(L, ['postpartum hemorrhage','pph']))
    df['grand_multipara'] = df['parity'].apply(lambda x: True if (pd.notna(x) and x>=5) else False) | df['social'].apply(lambda L: contains_any(L, ['grand multipara']))
    df['multiple_gestation'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ['multiple gestation','twin','twins','triplet']))
    df['polihydraminos_flag'] = df['liquor_flags'].apply(lambda L: contains_any(L, ['polyhydramnios','polihydramnios','polihydraminos']))
    df['placenta_abruption_flag'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ['placenta abruption','abruption']))
    df['placenta_prev_or_abruption'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ['placenta previa','placenta praevia','placenta abruption','abruption']))
    df['severe_anemia'] = df['hb_g_dl'].apply(lambda x: True if (pd.notna(x) and float(x)<7) else False)
    df['obstructed_prolonged'] = df['obstetric_history'].apply(lambda L: contains_any(L, ['obstructed','prolonged labor','prolonged']))
    # infer total_number_of_cs from obstetric_history when missing/zero
    df['inferred_cs'] = df['obstetric_history'].apply(lambda L: infer_total_cs_from_obstetric(L))
    df['total_number_of_cs'] = df.apply(lambda r: int(r['inferred_cs']) if (pd.isna(r.get('total_number_of_cs')) or int(r.get('total_number_of_cs') or 0)==0) and r['inferred_cs']>0 else int(r.get('total_number_of_cs') or 0), axis=1)
    return df

def coerce_numeric(df):
    numeric_cols = ['age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','labor_duration_hours']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan
    return df

def compute_target(df):
    # Use blood_loss >=1000 or blood_transfusion flag as PPH target when present.
    if 'blood_loss' not in df.columns:
        df['blood_loss'] = np.nan
    if 'blood_transfusion' not in df.columns:
        df['blood_transfusion'] = False
    df['is_pph'] = 0
    try:
        df.loc[df['blood_loss'].astype(float) >= 1000, 'is_pph'] = 1
    except Exception:
        pass
    df.loc[df['blood_transfusion'].astype(str).str.lower().isin(['true','1','yes','y']), 'is_pph'] = 1
    df['is_pph'] = df['is_pph'].astype(int)
    return df

# ---------- Pipeline builders ----------
def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)], remainder='passthrough')
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', max_features='sqrt', n_jobs=-1)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_prob = None
    if hasattr(pipeline, 'predict_proba'):
        try:
            probs = pipeline.predict_proba(X_test)
            if probs.ndim==2 and probs.shape[1]>1:
                y_prob = probs[:,1]
        except Exception:
            y_prob = None
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, zero_division=0),
    }
    if y_prob is not None and len(np.unique(y_test))>1:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_prob))
        except Exception:
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None
    metrics['y_prob_available'] = bool(y_prob is not None)
    metrics['y_test_unique_values'] = list(map(int, np.unique(y_test)))
    return metrics

# ---------- Management Command ----------
class Command(BaseCommand):
    help = 'Train PPH predictor from Patient table and save artifacts'

    def add_arguments(self, parser):
        parser.add_argument('--exclude-high-confidence', type=int, default=80)
        parser.add_argument('--test-size', type=float, default=0.2)
        parser.add_argument('--random-state', type=int, default=42)

    def handle(self, *args, **options):
        if Patient is None:
            self.stdout.write(self.style.ERROR('Patient model not importable. Run inside Django.'))
            return
        self.stdout.write('Loading patients...')
        df = load_patients_df()
        n_total = len(df)
        self.stdout.write(f'Total rows: {n_total}')
        df = normalize_lists(df)
        df = derive_flags(df)
        df = coerce_numeric(df)
        df = compute_target(df)

        # Apply deterministic rules
        df['pph_rule_flag'] = False
        df['pph_rule_weight'] = 0
        df['pph_rule_reasons'] = None
        for idx, row in df.iterrows():
            matched, weight, reasons = pph_deterministic_check(row.to_dict())
            df.at[idx, 'pph_rule_flag'] = matched
            df.at[idx, 'pph_rule_weight'] = weight
            df.at[idx, 'pph_rule_reasons'] = reasons

        exclude_thr = options['exclude_high_confidence']
        train_mask = ~((df['pph_rule_flag']) & (df['pph_rule_weight'] >= exclude_thr))
        train_df = df[train_mask].copy()
        n_train = len(train_df)
        self.stdout.write(f'Training rows after exclusion: {n_train} (excluded {n_total - n_train})')

        numeric_features = ['age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','labor_duration_hours']
        categorical_features = ['type_of_labor','mode_of_delivery','placenta_location','rupture_duration_hour','type_of_cs','perineum_integrity','instrumental_delivery']
        binary_features = ['history_pph','grand_multipara','multiple_gestation','polihydraminos_flag','placenta_prev_or_abruption','placenta_abruption_flag','severe_anemia','history_transfusion','obstructed_prolonged']

        # Ensure features exist
        for f in numeric_features + categorical_features + binary_features:
            if f not in train_df.columns:
                train_df[f] = np.nan

        X = train_df[numeric_features + categorical_features + binary_features].copy()
        # binarize
        for b in binary_features:
            X[b] = X[b].apply(lambda v: 1 if (str(v).lower() in ['true','1','yes'] or v is True) else 0)

        y = train_df['is_pph'].astype(int)
        valid_mask = ~train_df['is_pph'].isna()
        X = X[valid_mask]
        y = y[valid_mask]

        # split
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=options['test_size'], random_state=options['random_state'], stratify=y)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=options['test_size'], random_state=options['random_state'])

        self.stdout.write(f'Training on {len(X_train)} rows; testing on {len(X_test)} rows')

        pipeline = build_pipeline(numeric_features, categorical_features)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)
        metrics['n_total'] = int(n_total)
        metrics['n_train'] = int(len(X_train))
        metrics['n_test'] = int(len(X_test))
        metrics['n_trainable'] = int(len(X))

        artifacts_dir = Path(settings.BASE_DIR) / 'artifacts'
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifacts_dir / 'pph_pipeline.joblib'
        metrics_path = artifacts_dir / 'pph_metrics.json'
        featimp_path = artifacts_dir / 'pph_feature_importances.csv'
        rule_path = artifacts_dir / 'pph_rule_summary.json'

        joblib.dump(pipeline, model_path)
        with open(metrics_path, 'w') as fh:
            json.dump(metrics, fh, indent=2)
        # feature importances
        try:
            preproc = pipeline.named_steps['preprocessor']
            ohe = preproc.named_transformers_['cat'].named_steps['ohe']
            cat_names = ohe.get_feature_names_out(categorical_features)
            all_names = list(numeric_features) + list(cat_names) + list(binary_features)
            importances = pipeline.named_steps['classifier'].feature_importances_
            fi = sorted(zip(all_names, importances), key=lambda x: x[1], reverse=True)
            fi_df = pd.DataFrame(fi, columns=['feature','importance'])
            fi_df.to_csv(featimp_path, index=False)
        except Exception:
            pd.DataFrame([]).to_csv(featimp_path, index=False)

        # save rule summary
        rule_summary = {'rules': [{'weight': w, 'reason': r} for (_,w,r) in PPH_RULES]}
        with open(rule_path, 'w') as fh:
            json.dump(rule_summary, fh, indent=2)

        self.stdout.write(self.style.SUCCESS(f"Saved pipeline to {model_path}"))
        self.stdout.write(self.style.SUCCESS(f"Saved metrics to {metrics_path}"))
        self.stdout.write(self.style.SUCCESS('Training complete.'))
