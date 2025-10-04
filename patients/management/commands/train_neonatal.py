"""
Django management command to train Neonatal Complications predictor.

Placement: patients/management/commands/train_neonatal.py

- Implements deterministic clinical rules (weights & reasons) for high-risk neonatal outcomes.
- Trains RandomForest pipeline on requested features when rules do not decisively predict outcome.
- Saves artifacts to <BASE_DIR>/artifacts/neonatal_*.joblib/.json/.csv
- Fix: robust handling when 'parity' (or other expected columns) is missing from the Patient queryset.
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

# allow static editing outside Django
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

# ---------------- Deterministic neonatal rules (weights & reasons) ----------------
NEO_RULES = [
    (lambda r: r.get('preterm_birth_less_37_weeks', False), 50, "Preterm birth <37 weeks -> high NICU/HIE/death risk"),
    (lambda r: (str(r.get('ctg_category','')).lower().find('category_iii')>=0 or str(r.get('ctg_category','')).lower().find('category iii')>=0), 85, "CTG Category III -> high risk (HIE/NICU)"),
    (lambda r: (str(r.get('ctg_category','')).lower().find('category_ii')>=0 or str(r.get('ctg_category','')).lower().find('category ii')>=0), 30, "CTG Category II -> suspicious (NICU risk)"),
    (lambda r: r.get('placenta_abruption_flag', False), 60, "Placental abruption -> acute fetal hypoxia risk"),
    (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['placenta previa','placenta praevia','placenta abruption','abruption']), 25, "Placenta previa/abruption -> antepartum bleeding / preterm risk"),
    (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['multiple gestation','twin','twins','triplet']), 25, "Multiple gestation -> prematurity / NICU risk"),
    (lambda r: contains_any(r.get('current_pregnancy_fetal', []), ['breech','non-cephalic','transverse','oblique']), 15, "Non-cephalic presentation -> operative delivery / trauma risk"),
    (lambda r: (str(r.get('rupture_duration_hour','')).lower().find('18')>=0 or str(r.get('rupture_duration_hour','')).lower().find('24')>=0), 12, "Prolonged rupture (18-24h) -> infection / sepsis risk"),
    (lambda r: contains_any(r.get('indication_of_induction', []), ['prelabor_rupture_of_membranes_prom','prelabor rupture','prom']), 8, "PROM -> infection risk"),
    (lambda r: contains_any(r.get('obstetric_history', []), ['preeclampsia','pre-eclampsia','pre eclampsia']), 20, "History of preeclampsia -> uteroplacental insufficiency"),
    (lambda r: contains_any(r.get('menternal_medical', []), ['chronic hypertension','hypertension']), 12, "Chronic hypertension -> SGA/NICU risk"),
    (lambda r: contains_any(r.get('menternal_medical', []), ['diabetes','gdm','gestational diabetes']), 12, "Diabetes -> macrosomia/hypoglycemia/respiratory distress"),
    (lambda r: (not pd.isna(r.get('hb_g_dl')) and float(r.get('hb_g_dl',999)) < 7), 8, "Severe maternal anemia (Hb<7) -> fetal compromise risk"),
    (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['pregnancy post ivf','ivf','icsi']), 12, "Pregnancy post IVF/ICSI -> prematurity / NICU risk"),
    (lambda r: contains_any(r.get('current_pregnancy_fetal', []), ['iugr','intrauterine growth restriction','sga']), 20, "IUGR -> SGA/NICU risk"),
    (lambda r: contains_any(r.get('obstetric_history', []), ['stillbirth','neonatal death']), 30, "Prior stillbirth/neonatal death -> higher risk"),
    (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['polihydraminos','polyhydramnios','polihydramnios']), 12, "Polyhydramnios -> anomalies/cord prolapse risk"),
    (lambda r: (r.get('total_number_of_cs',0) > 1), 8, "Multiple prior CS (>1) -> complications risk"),
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

# ---------------- Data load & preprocessing ----------------
def load_patients_to_df(qs=None):
    if qs is None:
        if Patient is None:
            raise RuntimeError("Patient model not importable. Run inside Django.")
        qs = Patient.objects.all()
    rows = []
    # fields to pull
    fields = [
        'id','age','mode_of_delivery','instrumental_delivery','labor_duration_hours','rupture_duration_hour',
        'indication_of_induction','membrane_status','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','ctg_category','fetus_number',
        'congenital_anomalies','neonatal_death','nicu_admission','hie','birth_injuries','apgar_score','preterm_birth_less_37_weeks','total_number_of_cs','parity'
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
    return pd.DataFrame(rows)

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
    # Ensure parity exists to avoid KeyError (some DB exports may omit it)
    if 'parity' not in df.columns:
        df['parity'] = np.nan

    # Ensure 'social' exists and is normalized (normalize_list_columns should already have run)
    if 'social' not in df.columns:
        df['social'] = [[] for _ in range(len(df))]

    df['preterm_birth_less_37_weeks'] = df.get('preterm_birth_less_37_weeks', False).apply(lambda v: True if (str(v).lower() in ['true','1','yes'] or v is True or v==1) else False)
    df['placenta_abruption_flag'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ['placenta abruption','abruption']))
    df['multiple_gestation'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ['multiple gestation','twin','twins','triplet']))
    df['non_cephalic'] = df['current_pregnancy_fetal'].apply(lambda L: contains_any(L, ['breech','non-cephalic','transverse','oblique']))
    df['iugr_flag'] = df['current_pregnancy_fetal'].apply(lambda L: contains_any(L, ['iugr','intrauterine growth restriction','sga']))
    df['ivf_flag'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ['pregnancy post ivf','ivf','icsi']))
    df['polihydraminos_flag'] = df['liquor_flags'].apply(lambda L: contains_any(L, ['polyhydramnios','polihydramnios','polihydraminos']))
    df['history_still_neonatal_death'] = df['obstetric_history'].apply(lambda L: contains_any(L, ['stillbirth','neonatal death']))
    # grand multipara: check social list OR parity if available
    df['grand_multipara'] = df['social'].apply(lambda L: contains_any(L, ['grand multipara'])) | df['parity'].apply(lambda x: True if (pd.notna(x) and x>=5) else False)
    return df

def normalize_numeric_columns(df):
    numeric_cols = ['age','labor_duration_hours','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','apgar_score','parity']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan
    return df

def compute_target_neonatal(df):
    # composite adverse neonatal outcome: NICU admission, HIE, neonatal death, birth injury
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

def save_artifacts(pipeline, metrics, artifacts_dir=None, prefix='neonatal'):
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

class Command(BaseCommand):
    help = 'Train Neonatal complications predictor and save artifacts'

    def add_arguments(self, parser):
        parser.add_argument('--test-size', type=float, default=0.2)
        parser.add_argument('--random-state', type=int, default=42)
        parser.add_argument('--exclude-high-confidence', type=int, default=80)

    def handle(self, *args, **options):
        if Patient is None:
            self.stdout.write(self.style.ERROR('Patient model not importable. Run this command inside Django.'))
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

        # apply deterministic rules
        self.stdout.write('Applying deterministic neonatal rules...')
        df['neo_rule_flag'] = False
        df['neo_rule_weight'] = 0
        df['neo_rule_reasons'] = None
        for idx, row in df.iterrows():
            matched, weight, reasons = neonatal_deterministic_check(row.to_dict())
            df.at[idx, 'neo_rule_flag'] = matched
            df.at[idx, 'neo_rule_weight'] = weight
            df.at[idx, 'neo_rule_reasons'] = reasons

        exclude_thr = options['exclude_high_confidence']
        train_mask = ~((df['neo_rule_flag']) & (df['neo_rule_weight'] >= exclude_thr))
        train_df = df[train_mask].copy()
        n_total = len(df)
        n_train = len(train_df)
        self.stdout.write(f'Total rows: {n_total}. Trainable after exclusion: {n_train}')
        if n_train < 10:
            self.stdout.write(self.style.WARNING('Very few rows left for training after exclusion; proceeding anyway.'))

        global numeric_features, categorical_features, binary_features
        numeric_features = ['age','labor_duration_hours','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','apgar_score']
        categorical_features = ['mode_of_delivery','instrumental_delivery','rupture_duration_hour','membrane_status','ctg_category','fetus_number']
        binary_features = ['chronic_hypertension','diabetes_any','pre_eclampsia','severe_anemia','total_number_of_cs_gt1','grand_multipara','preterm_birth_less_37_weeks','placenta_prev_or_abruption','polihydraminos_flag','multiple_gestation','non_cephalic','iugr_flag','ivf_flag','history_still_neonatal_death','congenital_anomalies']

        train_df['diabetes_any'] = train_df['menternal_medical'].apply(lambda L: contains_any(L, ['diabetes','gdm','gestational diabetes']))
        train_df['chronic_hypertension'] = train_df['menternal_medical'].apply(lambda L: contains_any(L, ['chronic hypertension','hypertension']))
        train_df['pre_eclampsia'] = train_df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ['pre-eclampsia','preeclampsia','pre eclampsia']))
        train_df['total_number_of_cs_gt1'] = train_df['total_number_of_cs'].apply(lambda x: 1 if (pd.notna(x) and int(x)>1) else 0)

        X_all, missing_summary = build_feature_dataframe(train_df, numeric_features + categorical_features + binary_features)
        self.stdout.write('Missing values per feature: ' + json.dumps(missing_summary))

        X = X_all.copy()
        for b in binary_features:
            X[b] = X[b].apply(lambda v: 1 if (str(v).lower() in ['true','1','yes'] or v is True or v==1) else 0)

        y = train_df['target_neonatal_adverse'].astype(int)
        valid_mask = ~train_df['target_neonatal_adverse'].isna()
        X = X[valid_mask]; y = y[valid_mask]

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=options['test_size'], random_state=options['random_state'], stratify=y)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=options['test-size'], random_state=options['random-state'])

        self.stdout.write('Building pipeline and fitting model...')
        pipeline = build_pipeline(numeric_features, categorical_features)
        pipeline.fit(X_train, y_train)

        self.stdout.write('Evaluating model...')
        metrics = evaluate_model(pipeline, X_test, y_test)
        metrics['n_total'] = int(n_total)
        metrics['n_train'] = int(n_train)
        metrics['n_used_for_training'] = int(len(X_train))
        self.stdout.write('Training metrics:')
        self.stdout.write(json.dumps(metrics, indent=2))

        self.stdout.write('Saving artifacts...')
        artifacts = save_artifacts(pipeline, metrics, artifacts_dir=None, prefix='neonatal')
        self.stdout.write(self.style.SUCCESS(f"Saved pipeline to: {artifacts['model_path']}"))
        self.stdout.write(self.style.SUCCESS(f"Saved metrics to: {artifacts['metrics_path']}"))
        self.stdout.write(self.style.SUCCESS('Training complete.'))
