"""
Django management command: train_pph_direct

Trains a PPH (postpartum hemorrhage) predictor using the requested direct feature set.

Behavior:
- Applies deterministic clinical rules first (high-confidence overrides).
- Builds a RandomForest pipeline with numeric imputation and categorical one-hot encoding.
- Validates minimal coverage of direct features (configurable via --min-coverage).
- Saves artifacts to artifacts/pph_direct_pipeline.joblib and metrics JSON/feature-importances CSV.

Place as: patients/management/commands/train_pph_direct.py
"""
import os
import json
import re
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix

try:
    from patients.models import Patient
except Exception:
    Patient = None

# ---------------- Helpers ----------------
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
        if any(k.lower() in s for s in low):
            return True
    return False

# ---------------- Deterministic PPH rules (weights & reasons) ----------------
PPH_RULES = [
    (lambda r: contains_any(r.get('obstetric_history', []), ['postpartum hemorrhage','pph']), 20, "History of PPH (prior)"),
    (lambda r: ( (isinstance(r.get('social', []), list) and contains_any(r.get('social', []), ['grand multipara'])) or (r.get('parity') is not None and r.get('parity') >= 5) ), 12, "Grand multipara (>=5)"),
    (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['multiple gestation','twin','twins','triplet']), 12, "Multiple gestation"),
    (lambda r: (not pd.isna(r.get('estimated_fetal_weight_by_gm')) and float(r.get('estimated_fetal_weight_by_gm',0)) > 4000), 10, "Estimated fetal weight >=4000g"),
    (lambda r: contains_any(r.get('liquor_flags', []), ['polyhydramnios','polihydramnios','polihydraminos']), 9, "Polyhydramnios"),
    (lambda r: r.get('placenta_abruption_flag', False) or contains_any(r.get('current_pregnancy_menternal', []), ['placenta abruption','abruption']), 12, "Placental abruption"),
    (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['placenta previa','placenta praevia']), 15, "Placenta previa"),
    (lambda r: (not pd.isna(r.get('hb_g_dl')) and float(r.get('hb_g_dl',999)) < 7), 5, "Severe anemia (Hb<7) - morbidity risk"),
    (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['pre-eclampsia','preeclampsia','pre eclampsia']), 7, "Pre-eclampsia"),
    (lambda r: contains_any(r.get('obstetric_history', []), ['obstructed','prolonged labor','prolonged']), 7, "Obstructed/prolonged labor"),
    (lambda r: (r.get('total_number_of_cs',0) > 1), 7, "Multiple prior CS (>1)"),
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
    return (len(matches) > 0), total, reasons

# ---------------- Data extraction ----------------
def load_patients_to_df(qs=None):
    if qs is None:
        if Patient is None:
            raise RuntimeError("Patient model not importable. Run inside Django.")
        qs = Patient.objects.all()
    rows = []
    fields = [
        'id','age','parity','bmi','height','weight','total_number_of_cs','mode_of_delivery','type_of_labor',
        'labor_duration_hours','rupture_duration_hour','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l',
        'placenta_location','blood_loss','blood_transfusion','perineum_integrity','instrumental_delivery','type_of_cs'
    ]
    list_fields = ['menternal_medical','obstetric_history','current_pregnancy_menternal','social','liquor']
    for p in qs:
        d = {}
        for f in fields:
            try:
                d[f] = getattr(p, f)
            except Exception:
                d[f] = None
        for lf in list_fields:
            try:
                d[lf] = getattr(p, lf)
            except Exception:
                d[lf] = None
        rows.append(d)
    df = pd.DataFrame(rows)
    return df

# ---------------- Preprocessing & flags ----------------
def normalize_list_columns(df):
    list_cols = ['menternal_medical','obstetric_history','current_pregnancy_menternal','social','liquor']
    for c in list_cols:
        if c not in df.columns:
            df[c] = [[] for _ in range(len(df))]
        else:
            df[c] = df[c].apply(safe_parse_list)
    df['liquor_flags'] = df['liquor'].apply(lambda L: [s.lower() for s in L if isinstance(s, str)])
    return df

def derive_pph_flags(df):
    if 'parity' not in df.columns:
        df['parity'] = np.nan
    df['history_pph'] = df['obstetric_history'].apply(lambda L: contains_any(L, ['postpartum hemorrhage','pph']))
    df['grand_multipara'] = df['social'].apply(lambda L: contains_any(L, ['grand multipara'])) | df['parity'].apply(lambda x: True if (pd.notna(x) and x>=5) else False)
    df['multiple_gestation'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ['multiple gestation','twin','twins','triplet']))
    df['polihydraminos_flag'] = df['liquor_flags'].apply(lambda L: contains_any(L, ['polyhydramnios','polihydramnios','polihydraminos']))
    df['placenta_abruption_flag'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ['placenta abruption','abruption']))
    df['placenta_prev_or_abruption'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ['placenta previa','placenta praevia','placenta abruption','abruption']))
    df['severe_anemia'] = df.get('hb_g_dl', np.nan).apply(lambda x: True if (pd.notna(x) and float(x) < 7) else False)
    return df

def normalize_numeric_columns(df):
    numeric_cols = ['age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','labor_duration_hours']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan
    return df

def compute_target_pph(df):
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

# ---------------- Features & pipeline ----------------
def build_feature_dataframe(df, features):
    X = df.copy()
    missing_summary = {}
    for f in features:
        if f not in X.columns:
            X[f] = np.nan
        missing_summary[f] = int(X[f].isna().sum())
    return X[features], missing_summary

def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)], remainder='passthrough')
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_prob = None
    if hasattr(pipeline, 'predict_proba'):
        try:
            probs = pipeline.predict_proba(X_test)
            if probs.ndim ==2 and probs.shape[1] >1:
                y_prob = probs[:,1]
            else:
                try:
                    clf = pipeline.named_steps.get('classifier', None)
                    if clf is not None and hasattr(clf,'classes_') and len(clf.classes_)==1:
                        y_prob = np.ones(len(X_test)) if clf.classes_[0]==1 else np.zeros(len(X_test))
                except Exception:
                    y_prob = None
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

def save_artifacts(pipeline, metrics, artifacts_dir=None, prefix='pph_direct'):
    if artifacts_dir is None:
        artifacts_dir = os.path.join(getattr(settings,'BASE_DIR','.'),'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, f'{prefix}_pipeline.joblib')
    metrics_path = os.path.join(artifacts_dir, f'{prefix}_metrics.json')
    featimp_path = os.path.join(artifacts_dir, f'{prefix}_feature_importances.csv')
    joblib.dump(pipeline, model_path)
    with open(metrics_path, 'w') as fh:
        json.dump(metrics, fh, indent=2, default=lambda x: (x.tolist() if isinstance(x,np.ndarray) else x))
    try:
        preproc = pipeline.named_steps['preprocessor']
        try:
            feature_names = list(preproc.get_feature_names_out())
        except Exception:
            ohe = preproc.named_transformers_['cat'].named_steps['ohe']
            cat_names = list(ohe.get_feature_names_out())
            feature_names = list(numeric_features) + cat_names + list(binary_features)
        importances = pipeline.named_steps['classifier'].feature_importances_
        fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
        fi_df.to_csv(featimp_path, index=False)
    except Exception:
        pd.DataFrame([]).to_csv(featimp_path, index=False)
    return {'model_path': model_path, 'metrics_path': metrics_path, 'featimp_path': featimp_path}

def validate_min_coverage(df, columns, min_fraction=0.7):
    """
    Checks if enough non-null data is available for each feature column.
    Returns (ok, report_dict)
    """
    report = {}
    ok = True

    for col in columns:
        non_null_fraction = df[col].notnull().mean()
        report[col] = round(non_null_fraction, 3)
        if non_null_fraction < min_fraction:
            ok = False

    return ok, report

# ---------------- Django Command ----------------
class Command(BaseCommand):
    help = 'Train PPH predictor (direct feature set)'

    def add_arguments(self, parser):
        parser.add_argument('--test-size', type=float, default=0.2)
        parser.add_argument('--random-state', type=int, default=42)
        parser.add_argument('--min-coverage', type=float, default=0.05, help='Min fraction of non-missing for any required field to allow training.')

    def handle(self, *args, **options):
        if Patient is None:
            self.stdout.write(self.style.ERROR('Patient model not importable. Run inside Django.'))
            return

        self.stdout.write('Loading patients...')
        df = load_patients_to_df()

        self.stdout.write('Normalizing list columns...')
        df = normalize_list_columns(df)

        self.stdout.write('Deriving flags...')
        df = derive_pph_flags(df)

        self.stdout.write('Normalizing numeric columns...')
        df = normalize_numeric_columns(df)

        self.stdout.write('Computing PPH target...')
        df = compute_target_pph(df)

        # deterministic rules
        self.stdout.write('Applying deterministic PPH rules...')
        df['pph_rule_flag'] = False
        df['pph_rule_weight'] = 0
        df['pph_rule_reasons'] = None
        for idx, row in df.iterrows():
            matched, weight, reasons = pph_deterministic_check(row.to_dict())
            df.at[idx, 'pph_rule_flag'] = matched
            df.at[idx, 'pph_rule_weight'] = weight
            df.at[idx, 'pph_rule_reasons'] = reasons

        # features requested
        global numeric_features, categorical_features, binary_features
        numeric_features = ['age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','labor_duration_hours']
        categorical_features = ['type_of_labor','mode_of_delivery','placenta_location','rupture_duration_hour','type_of_cs','perineum_integrity','instrumental_delivery']
        binary_features = ['history_pph','grand_multipara','multiple_gestation','polihydraminos_flag','placenta_prev_or_abruption','placenta_abruption_flag','severe_anemia','history_transfusion','obstructed_prolonged']

        # create/ensure binary derived columns exist
        df['history_transfusion'] = df['menternal_medical'].apply(lambda L: contains_any(L, ['blood transfusion','transfusion'])) if 'menternal_medical' in df.columns else False
        df['obstructed_prolonged'] = df['obstetric_history'].apply(lambda L: contains_any(L, ['obstructed','prolonged labor','prolonged'])) if 'obstetric_history' in df.columns else False

        # Validate minimal coverage for training
        ok, report = validate_min_coverage(df, numeric_features + categorical_features, min_fraction=options['min_coverage'])
        self.stdout.write('Coverage report (per required field):')
        self.stdout.write(json.dumps(report, indent=2))
        if not ok:
            self.stdout.write(self.style.ERROR('Not enough coverage in direct fields to proceed with training. Increase data or lower --min-coverage'))
            return

        X_all, missing_summary = build_feature_dataframe(df, numeric_features + categorical_features + binary_features)
        self.stdout.write('Missing values per feature: ' + json.dumps(missing_summary))

        X = X_all.copy()
        for b in binary_features:
            X[b] = X[b].apply(lambda v: 1 if (str(v).lower() in ['true','1','yes'] or v is True or v==1) else 0)

        y = df['is_pph'].astype(int)
        valid_mask = ~df['is_pph'].isna()
        X = X[valid_mask]; y = y[valid_mask]

        if len(X) < 20:
            self.stdout.write(self.style.ERROR('Too few labelled rows to train (need >=20). Aborting.'))
            return

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=options['test_size'], random_state=options['random_state'], stratify=y)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=options['test-size'], random_state=options['random-state'])

        self.stdout.write('Building and fitting pipeline...')
        pipeline = build_pipeline(numeric_features, categorical_features)
        pipeline.fit(X_train, y_train)

        self.stdout.write('Evaluating model...')
        metrics = evaluate_model(pipeline, X_test, y_test)
        metrics['n_total'] = int(len(df))
        metrics['n_used_for_training'] = int(len(X_train))
        self.stdout.write('Training metrics:')
        self.stdout.write(json.dumps(metrics, indent=2))

        self.stdout.write('Saving artifacts...')
        artifacts = save_artifacts(pipeline, metrics, artifacts_dir=None, prefix='pph_direct')
        self.stdout.write(self.style.SUCCESS(f"Saved pipeline to: {artifacts['model_path']}"))
        self.stdout.write(self.style.SUCCESS(f"Saved metrics to: {artifacts['metrics_path']}"))
        self.stdout.write(self.style.SUCCESS('Training complete.'))
