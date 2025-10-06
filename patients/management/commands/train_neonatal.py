"""
Django management command: train_neonatal_direct

Trains a neonatal adverse-outcome classifier using the "direct" feature set requested.

Behavior:
- Applies deterministic clinical rules first (high-confidence overrides).
- Builds a RandomForest pipeline with numeric imputation and categorical one-hot encoding.
- Validates that at least a minimal subset of direct numeric/categorical fields exist with non-missing values,
  otherwise aborts and prints a clear message.
- Saves artifacts to artifacts/neonatal_direct_pipeline.joblib and metrics JSON/feature-importances CSV.

Place as: patients/management/commands/train_neonatal_direct.py
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

# Try import Patient model
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

# ---------------- Deterministic rules (weights & reasons) ----------------
NEO_RULES = [
    (lambda r: r.get('preterm_birth_less_37_weeks', False), 60, "Preterm birth <37 weeks"),
    (lambda r: (str(r.get('ctg_category','')).lower().find('category_iii')>=0), 85, "CTG Category III"),
    (lambda r: (str(r.get('ctg_category','')).lower().find('category_ii')>=0), 35, "CTG Category II"),
    (lambda r: r.get('placenta_abruption_flag', False), 60, "Placental abruption"),
    (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['placenta previa','placenta praevia','placenta abruption','abruption']), 30, "Placenta previa/abruption"),
    (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['multiple gestation','twin','twins','triplet']), 30, "Multiple gestation"),
    (lambda r: contains_any(r.get('current_pregnancy_fetal', []), ['breech','non-cephalic','transverse','oblique']), 15, "Non-cephalic presentation"),
    (lambda r: contains_any(r.get('liquor_flags', []), ['polyhydramnios','polihydramnios','polihydraminos']), 12, "Polyhydramnios"),
    (lambda r: contains_any(r.get('obstetric_history', []), ['preeclampsia','pre-eclampsia','pre eclampsia']), 25, "History of preeclampsia"),
    (lambda r: contains_any(r.get('menternal_medical', []), ['chronic hypertension','hypertension']), 15, "Chronic hypertension"),
    (lambda r: contains_any(r.get('menternal_medical', []), ['diabetes','gdm','gestational diabetes']), 12, "Diabetes"),
    (lambda r: (not pd.isna(r.get('hb_g_dl')) and float(r.get('hb_g_dl',999)) < 7), 8, "Severe anemia (Hb<7)"),
    (lambda r: contains_any(r.get('current_pregnancy_fetal', []), ['iugr','intrauterine growth restriction','sga']), 20, "IUGR"),
    (lambda r: contains_any(r.get('obstetric_history', []), ['stillbirth','neonatal death']), 30, "Prior stillbirth/neonatal death"),
    (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['pregnancy post ivf','ivf','icsi']), 12, "Pregnancy post IVF/ICSI"),
]

def neonatal_deterministic_check(rowdict):
    matches = []
    total = 0
    reasons = []
    for cond, weight, reason in NEO_RULES:
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
        'id','age','parity','mode_of_delivery','instrumental_delivery','labor_duration_hours','rupture_duration_hour',
        'indication_of_induction','membrane_status','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','ctg_category','fetus_number',
        'congenital_anomalies','neonatal_death','nicu_admission','hie','birth_injuries','apgar_score','preterm_birth_less_37_weeks','total_number_of_cs','placenta_location'
    ]
    list_fields = ['menternal_medical','obstetric_history','current_pregnancy_menternal','current_pregnancy_fetal','social','liquor']
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

# ---------------- Preprocessing & feature derivation ----------------
def normalize_list_columns(df):
    list_cols = ['menternal_medical','obstetric_history','current_pregnancy_menternal','current_pregnancy_fetal','social','liquor']
    for c in list_cols:
        if c not in df.columns:
            df[c] = [[] for _ in range(len(df))]
        else:
            df[c] = df[c].apply(safe_parse_list)
    df['liquor_flags'] = df['liquor'].apply(lambda L: [s.lower() for s in L if isinstance(s, str)])
    return df

def derive_flags(df):
    # ensure parity exists
    if 'parity' not in df.columns:
        df['parity'] = np.nan
    df['preterm_birth_less_37_weeks'] = df.get('preterm_birth_less_37_weeks', False).apply(lambda v: True if (str(v).lower() in ['true','1','yes'] or v is True or v==1) else False)
    df['placenta_abruption_flag'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ['placenta abruption','abruption']))
    df['placenta_prev_or_abruption'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ['placenta previa','placenta praevia','placenta abruption','abruption']))
    df['multiple_gestation'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ['multiple gestation','twin','twins','triplet']))
    df['non_cephalic'] = df['current_pregnancy_fetal'].apply(lambda L: contains_any(L, ['breech','non-cephalic','transverse','oblique']))
    df['iugr_flag'] = df['current_pregnancy_fetal'].apply(lambda L: contains_any(L, ['iugr','intrauterine growth restriction','sga']))
    df['ivf_flag'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ['pregnancy post ivf','ivf','icsi']))
    df['polihydraminos_flag'] = df['liquor_flags'].apply(lambda L: contains_any(L, ['polyhydramnios','polihydramnios','polihydraminos']))
    df['history_still_neonatal_death'] = df['obstetric_history'].apply(lambda L: contains_any(L, ['stillbirth','neonatal death']))
    df['grand_multipara'] = df['social'].apply(lambda L: contains_any(L, ['grand multipara'])) | df['parity'].apply(lambda x: True if (pd.notna(x) and x>=5) else False)
    df['severe_anemia'] = df.get('hb_g_dl', np.nan).apply(lambda x: True if (pd.notna(x) and float(x) < 7) else False)
    return df

def normalize_numeric_columns(df):
    numeric_cols = ['age','labor_duration_hours','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','apgar_score','parity','total_number_of_cs']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan
    return df

def compute_target_neonatal(df):
    df['target_neonatal_adverse'] = 0
    try:
        df.loc[df['nicu_admission'].astype(str).str.lower().isin(['true','1','yes','y']), 'target_neonatal_adverse'] = 1
    except Exception:
        pass
    try:
        df.loc[df['hie'].astype(str).str.lower().isin(['true','1','yes','y']), 'target_neonatal_adverse'] = 1
    except Exception:
        pass
    try:
        df.loc[df['neonatal_death'].astype(str).str.lower().isin(['true','1','yes','y']), 'target_neonatal_adverse'] = 1
    except Exception:
        pass
    try:
        df.loc[df['birth_injuries'].astype(str).str.lower().isin(['true','1','yes','y']), 'target_neonatal_adverse'] = 1
    except Exception:
        pass
    df['target_neonatal_adverse'] = df['target_neonatal_adverse'].astype(int)
    return df

# ---------------- Validation helpers ----------------
def validate_min_coverage(df, required_fields, min_fraction=0.1):
    """
    Ensure at least min_fraction of rows have non-missing values for any of required_fields.
    Returns (ok, report). Here ok means at least some minimal coverage exists for model training.
    """
    report = {}
    n = len(df)
    ok = False
    for f in required_fields:
        non_missing = int(df[f].notna().sum()) if f in df.columns else 0
        frac = non_missing / n if n>0 else 0.0
        report[f] = {'present': f in df.columns, 'non_missing': non_missing, 'fraction': round(frac,3)}
        if frac >= min_fraction:
            ok = True
    return ok, report

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

def save_artifacts(pipeline, metrics, artifacts_dir=None, prefix='neonatal_direct'):
    if artifacts_dir is None:
        artifacts_dir = os.path.join(getattr(settings,'BASE_DIR','.'),'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, f'{prefix}_pipeline.joblib')
    metrics_path = os.path.join(artifacts_dir, f'{prefix}_metrics.json')
    featimp_path = os.path.join(artifacts_dir, f'{prefix}_feature_importances.csv')
    joblib.dump(pipeline, model_path)
    with open(metrics_path,'w') as fh:
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

# ---------------- Django Command ----------------
class Command(BaseCommand):
    help = 'Train neonatal adverse-outcome predictor (direct feature set)'

    def add_arguments(self, parser):
        parser.add_argument('--test-size', type=float, default=0.2)
        parser.add_argument('--random-state', type=int, default=42)
        parser.add_argument('--min-coverage', type=float, default=0.1, help='Min fraction of non-missing for any required field to allow training.')

    def handle(self, *args, **options):
        if Patient is None:
            self.stdout.write(self.style.ERROR('Patient model not importable. Run inside Django.'))
            return

        self.stdout.write('Loading patients...')
        df = load_patients_to_df()

        self.stdout.write('Normalizing list columns...')
        df = normalize_list_columns(df)

        self.stdout.write('Deriving flags...')
        df = derive_flags(df)

        self.stdout.write('Normalizing numeric columns...')
        df = normalize_numeric_columns(df)

        self.stdout.write('Computing neonatal target...')
        df = compute_target_neonatal(df)

        # apply deterministic rules (to optionally exclude high-confidence cases)
        self.stdout.write('Applying deterministic neonatal rules...')
        df['neo_rule_flag'] = False
        df['neo_rule_weight'] = 0
        df['neo_rule_reasons'] = None
        for idx, row in df.iterrows():
            matched, weight, reasons = neonatal_deterministic_check(row.to_dict())
            df.at[idx, 'neo_rule_flag'] = matched
            df.at[idx, 'neo_rule_weight'] = weight
            df.at[idx, 'neo_rule_reasons'] = reasons

        # features requested by user as "direct"
        global numeric_features, categorical_features, binary_features
        numeric_features = ['age','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','labor_duration_hours','apgar_score']
        categorical_features = ['ctg_category','mode_of_delivery','instrumental_delivery','membrane_status','rupture_duration_hour','fetus_number','placenta_location']
        binary_features = ['preterm_birth_less_37_weeks','placenta_prev_or_abruption','multiple_gestation','non_cephalic','iugr_flag','ivf_flag','history_still_neonatal_death','congenital_anomalies','grand_multipara','severe_anemia','chronic_hypertension','diabetes_any','pre_eclampsia']

        # Validate minimal coverage: at least one of the numeric_features should have min coverage (user-configurable)
        ok, report = validate_min_coverage(df, numeric_features + categorical_features, min_fraction=options['min_coverage'])
        self.stdout.write('Coverage report (per required field):')
        self.stdout.write(json.dumps(report, indent=2))
        if not ok:
            self.stdout.write(self.style.ERROR('Not enough coverage in direct fields to proceed with training. Increase data or lower --min-coverage'))
            return

        # build X,y
        X_all, missing_summary = build_feature_dataframe(df, numeric_features + categorical_features + binary_features)
        self.stdout.write('Missing values per feature: ' + json.dumps(missing_summary))

        X = X_all.copy()
        for b in binary_features:
            X[b] = X[b].apply(lambda v: 1 if (str(v).lower() in ['true','1','yes'] or v is True or v==1) else 0)

        y = df['target_neonatal_adverse'].astype(int)
        valid_mask = ~df['target_neonatal_adverse'].isna()
        X = X[valid_mask]; y = y[valid_mask]

        if len(X) < 20:
            self.stdout.write(self.style.ERROR('Too few labelled rows to train (need >=20). Aborting.'))
            return

        # train/test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=options['test_size'], random_state=options['random_state'], stratify=y)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=options['test_size'], random_state=options['random_state'])

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
        artifacts = save_artifacts(pipeline, metrics, artifacts_dir=None, prefix='neonatal_direct')
        self.stdout.write(self.style.SUCCESS(f"Saved pipeline to: {artifacts['model_path']}"))
        self.stdout.write(self.style.SUCCESS(f"Saved metrics to: {artifacts['metrics_path']}"))
        self.stdout.write(self.style.SUCCESS('Training complete.'))
