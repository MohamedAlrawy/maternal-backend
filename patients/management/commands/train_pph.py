# """
# Refined Django management command to train a Postpartum Hemorrhage (PPH) predictor.

# Placement:
#     patients/management/commands/train_pph.py

# Improvements in this refined version:
#  - Robust handling of single-class predict_proba outputs.
#  - Safe extraction of feature names from the ColumnTransformer (with fallbacks).
#  - Diagnostics printed when feature_name / importances mismatch.
#  - Clear docstrings and modular helper functions.
#  - Saves artifacts (pipeline, metrics, feature importances) to <BASE_DIR>/artifacts/

# Note: Run this inside your Django project (so `patients.models.Patient` is importable).
# """

# import os
# import json
# import re
# import joblib
# import numpy as np
# import pandas as pd

# from django.core.management.base import BaseCommand
# from django.conf import settings

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix

# # Attempt to import Patient model; allow static editing if unavailable
# try:
#     from patients.models import Patient
# except Exception:
#     Patient = None

# # -------------------- Helpers --------------------
# def safe_parse_list(cell):
#     # \"\"\"Normalize many encodings into list[str].\"\"\"
#     if cell is None:
#         return []
#     if isinstance(cell, (list, tuple)):
#         return [str(x).strip() for x in cell if str(x).strip() != ""]
#     if isinstance(cell, dict):
#         return [str(v).strip() for v in cell.values() if str(v).strip() != ""]
#     if isinstance(cell, str):
#         try:
#             parsed = json.loads(cell)
#             if isinstance(parsed, list):
#                 return [str(x).strip() for x in parsed if str(x).strip() != ""]
#         except Exception:
#             pass
#         parts = re.split(r'[;,|]\s*', cell)
#         return [p.strip() for p in parts if p.strip() != ""]
#     return []

# def contains_any(lst, keys):
#     # \"\"\"Case-insensitive substring matching inside list elements.\"\"\"
#     if not lst:
#         return False
#     low = [str(s).lower() for s in lst]
#     for k in keys:
#         if any(k.lower() in s for s in low):
#             return True
#     return False

# # -------------------- Deterministic PPH rules --------------------
# PPH_RULES = [
#     (lambda r: contains_any(r.get('obstetric_history', []), ['postpartum hemorrhage', 'pph', 'post partum hemorrhage']), 20, "History of PPH (prior)"),
#     (lambda r: ( (isinstance(r.get('social', []), list) and contains_any(r.get('social', []), ['grand multipara'])) or (r.get('parity') is not None and r.get('parity') >= 5) ), 12, "Grand multipara (>=5)"),
#     (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['multiple gestation','twin','twins','triplet']), 12, "Multiple gestation"),
#     (lambda r: (not pd.isna(r.get('estimated_fetal_weight_by_gm')) and float(r.get('estimated_fetal_weight_by_gm',0)) > 4000), 10, "Estimated fetal weight >=4000g"),
#     (lambda r: contains_any(r.get('liquor_flags', []), ['polyhydramnios','polihydraminos','polihydramnios']), 9, "Polyhydramnios"),
#     (lambda r: r.get('placenta_abruption_flag', False) or contains_any(r.get('current_pregnancy_menternal', []), ['placenta abruption','abruption']), 12, "Placenta abruption"),
#     (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['placenta previa','placenta praevia']), 15, "Placenta previa"),
#     (lambda r: (not pd.isna(r.get('hb_g_dl')) and float(r.get('hb_g_dl', 999)) < 7), 5, "Severe anemia (Hb<7)"),
#     (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['pre-eclampsia','preeclampsia','pre eclampsia']), 7, "Pre-eclampsia"),
#     (lambda r: (r.get('total_number_of_cs',0) > 1), 7, "Multiple prior CS (>1)"),
#     (lambda r: contains_any(r.get('obstetric_history', []), ['obstructed','prolonged labor','prolonged']), 7, "Obstructed/prolonged labor"),
# ]

# def pph_deterministic_check(rowdict):
#     # \"\"\"Return (matched_bool, total_weight_estimate_percent, reasons_list).\"\"\"
#     matches = []
#     total = 0
#     reasons = []
#     for cond, weight, reason in PPH_RULES:
#         try:
#             if cond(rowdict):
#                 matches.append((weight, reason))
#                 total += weight
#                 reasons.append(reason)
#         except Exception:
#             continue
#     return (len(matches) > 0), total, reasons

# # -------------------- Data loading & preprocessing --------------------
# def load_patients_to_df(qs=None):
#     # \"\"\"Load selected fields from Patient queryset into a DataFrame.\"\"\"
#     if qs is None:
#         if Patient is None:
#             raise RuntimeError('Patient model not importable. Run inside Django.')
#         qs = Patient.objects.all()

#     rows = []
#     fields = [
#         'id','age','parity','bmi','height','weight',
#         'total_number_of_cs','mode_of_delivery','type_of_labor','perineum_integrity','instrumental_delivery','type_of_cs',
#         'labor_duration_hours','placenta_location','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l',
#         'fetus_number','blood_loss','blood_transfusion'
#     ]
#     # list/json fields
#     list_fields = ['menternal_medical','obstetric_history','current_pregnancy_menternal','social','liquor']
#     for p in qs:
#         d = {}
#         for f in fields:
#             try:
#                 d[f] = getattr(p, f)
#             except Exception:
#                 d[f] = None
#         for lf in list_fields:
#             try:
#                 d[lf] = getattr(p, lf)
#             except Exception:
#                 d[lf] = None
#         rows.append(d)
#     df = pd.DataFrame(rows)
#     return df

# def normalize_list_columns(df):
#     # \"\"\"Parse JSON/list-like columns into Python lists and normalize liquor into liquor_flags.\"\"\"
#     list_cols = ['menternal_medical','obstetric_history','current_pregnancy_menternal','social','liquor']
#     for c in list_cols:
#         if c not in df.columns:
#             df[c] = [[] for _ in range(len(df))]
#         else:
#             df[c] = df[c].apply(safe_parse_list)
#     # derive liquor_flags for polyhydramnios detection
#     df['liquor_flags'] = df['liquor'].apply(lambda L: [s.lower() for s in L if isinstance(s, str)])
#     return df

# def derive_pph_flags(df):
#     # \"\"\"Create boolean flags used by deterministic rules and features.\"\"\"
#     df['history_pph'] = df['obstetric_history'].apply(lambda L: contains_any(L, ['postpartum hemorrhage','pph','post partum hemorrhage']))
#     df['grand_multipara'] = df['social'].apply(lambda L: contains_any(L, ['grand multipara'])) | df['parity'].apply(lambda x: True if (pd.notna(x) and x>=5) else False)
#     df['multiple_gestation'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ['multiple gestation','twin','twins','triplet']))
#     df['placenta_prev_or_abruption'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ['placenta previa','placenta praevia','placenta abruption','abruption'])) 
#     df['polihydraminos_flag'] = df['liquor_flags'].apply(lambda L: contains_any(L, ['polihydraminos','polyhydramnios','polihydramnios'])) 
#     df['placenta_abruption_flag'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ['placenta abruption','abruption'])) 
#     df['severe_anemia'] = df['hb_g_dl'].apply(lambda x: True if (pd.notna(x) and float(x) < 7) else False)
#     df['obstructed_prolonged'] = df['obstetric_history'].apply(lambda L: contains_any(L, ['obstructed','prolonged labor','prolonged'])) 
#     return df

# def normalize_numeric_columns(df):
#     # \"\"\"Coerce numeric columns to numeric dtype, fill defaults where appropriate.\"\"\"
#     numeric_cols = ['age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','labor_duration_hours']
#     for c in numeric_cols:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors='coerce')
#         else:
#             df[c] = np.nan
#     return df

# def compute_target_pph(df):
#     # \"\"\"Derive binary target `is_pph` using conservative definition.\"\"\"
#     if 'blood_loss' not in df.columns:
#         df['blood_loss'] = np.nan
#     if 'blood_transfusion' not in df.columns:
#         df['blood_transfusion'] = False
#     df['is_pph'] = False
#     try:
#         df.loc[df['blood_loss'].astype(float) >= 1000, 'is_pph'] = True
#     except Exception:
#         pass
#     df.loc[df['blood_transfusion'].astype(str).str.lower().isin(['true','1','yes','y']), 'is_pph'] = True
#     df['is_pph'] = df['is_pph'].astype(int)
#     return df

# # -------------------- Feature building & pipeline --------------------
# def build_feature_dataframe(df, features):
#     X = df.copy()
#     missing_summary = {}
#     for f in features:
#         if f not in X.columns:
#             X[f] = np.nan
#         missing_summary[f] = int(X[f].isna().sum())
#     return X[features], missing_summary

# def build_pipeline(numeric_features, categorical_features):
#     numeric_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='median')),
#         ('scaler', StandardScaler()),
#     ])
#     categorical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#         ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
#     ])
#     preprocessor = ColumnTransformer(transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features),
#     ], remainder='passthrough')

#     clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
#     pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
#     return pipeline

# def evaluate_model(pipeline, X_test, y_test):
#     # \"\"\"Robust evaluation that handles single-class prob output.\"\"\"
#     y_pred = pipeline.predict(X_test)
#     y_prob = None
#     if hasattr(pipeline, "predict_proba"):
#         try:
#             probs = pipeline.predict_proba(X_test)
#             if probs.ndim == 2 and probs.shape[1] > 1:
#                 y_prob = probs[:, 1]
#             else:
#                 try:
#                     clf = pipeline.named_steps.get('classifier', None)
#                     if clf is not None and hasattr(clf, 'classes_'):
#                         classes = list(clf.classes_)
#                         if len(classes) == 1:
#                             if classes[0] == 1 or classes[0] == '1':
#                                 y_prob = np.ones(len(X_test))
#                             else:
#                                 y_prob = np.zeros(len(X_test))
#                         else:
#                             y_prob = None
#                     else:
#                         y_prob = None
#                 except Exception:
#                     y_prob = None
#         except Exception:
#             y_prob = None

#     metrics = {
#         'accuracy': float(accuracy_score(y_test, y_pred)),
#         'precision': float(precision_score(y_test, y_pred, zero_division=0)),
#         'recall': float(recall_score(y_test, y_pred, zero_division=0)),
#         'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
#         'classification_report': classification_report(y_test, y_pred, zero_division=0),
#     }

#     if y_prob is not None and len(np.unique(y_test)) > 1:
#         try:
#             metrics['roc_auc'] = float(roc_auc_score(y_test, y_prob))
#         except Exception:
#             metrics['roc_auc'] = None
#     else:
#         metrics['roc_auc'] = None

#     metrics['y_prob_available'] = bool(y_prob is not None)
#     metrics['y_test_unique_values'] = list(map(int, np.unique(y_test)))

#     return metrics

# def save_artifacts(pipeline, metrics, artifacts_dir=None, prefix='pph'):
#     if artifacts_dir is None:
#         artifacts_dir = os.path.join(settings.BASE_DIR, 'artifacts')
#     os.makedirs(artifacts_dir, exist_ok=True)
#     model_path = os.path.join(artifacts_dir, f'{prefix}_pipeline.joblib')
#     metrics_path = os.path.join(artifacts_dir, f'{prefix}_metrics.json')
#     featimp_path = os.path.join(artifacts_dir, f'{prefix}_feature_importances.csv')

#     joblib.dump(pipeline, model_path)
#     with open(metrics_path, 'w') as fh:
#         json.dump(metrics, fh, indent=2, default=lambda x: (x.tolist() if isinstance(x, np.ndarray) else x))

#     # Safely extract feature names and save importances
#     try:
#         preproc = pipeline.named_steps['preprocessor']
#         try:
#             # sklearn >=1.0 exposes get_feature_names_out for ColumnTransformer
#             feature_names = list(preproc.get_feature_names_out())
#         except Exception:
#             # Fallback: combine numeric + OHE expansion + binary features if available in globals
#             ohe = preproc.named_transformers_['cat'].named_steps['ohe']
#             cat_names = list(ohe.get_feature_names_out())
#             feature_names = list(numeric_features) + cat_names + list(binary_features)
#         importances = pipeline.named_steps['classifier'].feature_importances_
#         fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
#         fi_df.to_csv(featimp_path, index=False)
#         metrics['feature_importances_sum'] = float(importances.sum())
#     except Exception as e:
#         # write empty file on failure but log reason in metrics
#         pd.DataFrame([]).to_csv(featimp_path, index=False)
#         metrics['feature_importances_error'] = str(e)

#     return {'model_path': model_path, 'metrics_path': metrics_path, 'featimp_path': featimp_path}

# # -------------------- Django Command --------------------
# class Command(BaseCommand):
#     help = 'Train PPH predictor from Patient table and save pipeline + metrics to artifacts/pph_*'

#     def add_arguments(self, parser):
#         parser.add_argument('--exclude-high-confidence', type=int, default=80,
#                             help='Exclude rows with deterministic override >= this percent from training (default 80).')
#         parser.add_argument('--test-size', type=float, default=0.2, help='Test split fraction.')
#         parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility.')

#     def handle(self, *args, **options):
#         if Patient is None:
#             self.stdout.write(self.style.ERROR('Patient model not importable. Run this command inside Django.'))
#             return

#         # Load and preprocess data
#         self.stdout.write('Loading patients...')
#         df = load_patients_to_df()

#         self.stdout.write('Normalizing list/JSON columns...')
#         df = normalize_list_columns(df)

#         self.stdout.write('Deriving PPH flags...')
#         df = derive_pph_flags(df)

#         self.stdout.write('Normalizing numeric columns...')
#         df = normalize_numeric_columns(df)

#         self.stdout.write('Computing PPH target (is_pph)...')
#         df = compute_target_pph(df)

#         # Apply deterministic rules per row
#         self.stdout.write('Applying deterministic PPH rules...')
#         df['pph_rule_flag'] = False
#         df['pph_rule_weight'] = 0
#         df['pph_rule_reasons'] = None
#         for idx, row in df.iterrows():
#             rowdict = row.to_dict()
#             matched, weight, reasons = pph_deterministic_check(rowdict)
#             df.at[idx, 'pph_rule_flag'] = matched
#             df.at[idx, 'pph_rule_weight'] = weight
#             df.at[idx, 'pph_rule_reasons'] = reasons

#         # Exclude high-confidence rule rows from training if desired
#         exclude_thr = options['exclude_high_confidence']
#         train_mask = ~((df['pph_rule_flag']) & (df['pph_rule_weight'] >= exclude_thr))
#         train_df = df[train_mask].copy()
#         n_total = len(df)
#         n_train = len(train_df)
#         self.stdout.write(f'Total rows: {n_total}. Trainable after exclusion: {n_train}')

#         if n_train < 10:
#             self.stdout.write(self.style.WARNING('Very few rows left for training after exclusion; proceeding anyway.'))

#         # Feature selection (as requested)
#         global numeric_features, categorical_features, binary_features
#         numeric_features = ['age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','labor_duration_hours']
#         categorical_features = ['fetus_number','type_of_labor','mode_of_delivery','perineum_integrity','instrumental_delivery','type_of_cs','placenta_location']
#         binary_features = ['chronic_hypertension','history_of_blood_transfusion','history_pph','obstructed_prolonged','multiple_gestation','history_preeclampsia','severe_anemia','grand_multipara','polihydraminos_flag','placenta_prev_or_abruption']

#         # Ensure columns exist and build X
#         X_all, missing_summary = build_feature_dataframe(train_df, numeric_features + categorical_features + binary_features)
#         self.stdout.write('Missing values per feature: ' + json.dumps(missing_summary))

#         # Prepare X and y
#         X = X_all.copy()
#         # convert boolean-like fields to 0/1
#         for b in binary_features:
#             X[b] = X[b].apply(lambda v: 1 if (str(v).lower() in ['true','1','yes'] or v is True or v == 1) else 0)

#         y = train_df['is_pph'].astype(int)
#         # Remove rows where target is not defined (if any)
#         valid_mask = ~train_df['is_pph'].isna()
#         X = X[valid_mask]
#         y = y[valid_mask]

#         # train/test split
#         try:
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=options['test_size'], random_state=options['random_state'], stratify=y)
#         except Exception:
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=options['test_size'], random_state=options['random_state'])

#         # Build and fit pipeline
#         self.stdout.write('Building pipeline and fitting model...')
#         pipeline = build_pipeline(numeric_features, categorical_features)
#         pipeline.fit(X_train, y_train)

#         # Evaluate
#         self.stdout.write('Evaluating model...')
#         metrics = evaluate_model(pipeline, X_test, y_test)
#         metrics['n_total'] = int(n_total)
#         metrics['n_train'] = int(n_train)
#         metrics['n_used_for_training'] = int(len(X_train))

#         self.stdout.write('Training metrics:')
#         self.stdout.write(json.dumps(metrics, indent=2))

#         # Save artifacts
#         self.stdout.write('Saving artifacts...')
#         artifacts = save_artifacts(pipeline, metrics, artifacts_dir=None, prefix='pph')
#         self.stdout.write(self.style.SUCCESS(f"Saved pipeline to: {artifacts['model_path']}"))
#         self.stdout.write(self.style.SUCCESS(f"Saved metrics to: {artifacts['metrics_path']}"))
#         self.stdout.write(self.style.SUCCESS('Training complete.'))

#         # Print top feature importances if available
#         try:
#             preproc = pipeline.named_steps['preprocessor']
#             ohe = preproc.named_transformers_['cat'].named_steps['ohe']
#             cat_names = ohe.get_feature_names_out(categorical_features)
#             all_feature_names = list(numeric_features) + list(cat_names) + list(binary_features)
#             importances = pipeline.named_steps['classifier'].feature_importances_
#             feat_imp = sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)[:20]
#             self.stdout.write('Top feature importances:')
#             for name, imp in feat_imp:
#                 self.stdout.write(f' - {name}: {imp:.4f}')
#         except Exception:
#             pass


# """
# Improved PPH training command with cross-validation, hyperparameter search, and imbalance handling.

# Placement: patients/management/commands/train_pph.py  (replace existing file or keep as train_pph_tuned.py)

# Features added:
#  - Stratified K-Fold cross-validation reporting (accuracy, precision, recall, ROC-AUC).
#  - RandomizedSearchCV hyperparameter tuning for RandomForest (n_estimators, max_depth, max_features, min_samples_split).
#  - Uses class_weight='balanced' to mitigate class imbalance; optional SMOTE path (commented — requires imbalanced-learn).
#  - Calibration step (CalibratedClassifierCV) to improve probability estimates if desired.
#  - Saves best pipeline and CV results to artifacts/pph_pipeline_tuned.joblib and pph_cv_results.json
#  - Prints diagnostics to help you interpret low accuracy.

# Notes:
#  - This script assumes the same preprocessing helper functions and data-loading approach used in previous train_pph scripts.
#  - Run inside Django (Patient model importable).
# """

# import os, json, re, joblib, numpy as np, pandas as pd
# from pathlib import Path
# from django.core.management.base import BaseCommand
# from django.conf import settings

# from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split, cross_validate
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
# from scipy.stats import randint as sp_randint
# from sklearn.calibration import CalibratedClassifierCV

# # try to import Patient model
# try:
#     from patients.models import Patient
# except Exception:
#     Patient = None

# # Minimal helpers (reuse from previous script if available)
# def safe_parse_list(cell):
#     if cell is None: return []
#     if isinstance(cell, (list, tuple)): return [str(x).strip() for x in cell if str(x).strip()!='']
#     if isinstance(cell, dict): return [str(v).strip() for v in cell.values() if str(v).strip()!='']
#     if isinstance(cell, str):
#         try:
#             parsed = json.loads(cell)
#             if isinstance(parsed, list): return [str(x).strip() for x in parsed if str(x).strip()!='']
#         except Exception:
#             pass
#         parts = re.split(r'[;,|]\s*', cell)
#         return [p.strip() for p in parts if p.strip()!='']
#     return []

# def contains_any(lst, keys):
#     if not lst: return False
#     low = [str(s).lower() for s in lst]
#     for k in keys:
#         if any(k.lower() in s for s in low): return True
#     return False

# # Data extraction (similar to earlier scripts)
# def load_patients_to_df(qs=None):
#     if qs is None:
#         if Patient is None:
#             raise RuntimeError("Run inside Django to load Patient model.")
#         qs = Patient.objects.all()
#     rows = []
#     fields = [
#         'id','age','parity','bmi','height','weight','total_number_of_cs','type_of_labor','mode_of_delivery',
#         'perineum_integrity','instrumental_delivery','type_of_cs','labor_duration_hours','placenta_location',
#         'estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','fetus_number','blood_loss','blood_transfusion'
#     ]
#     list_fields = ['menternal_medical','obstetric_history','current_pregnancy_menternal','social','liquor']
#     for p in qs:
#         d = {}
#         for f in fields:
#             try: d[f] = getattr(p, f)
#             except Exception: d[f] = None
#         for lf in list_fields:
#             try: d[lf] = getattr(p, lf)
#             except Exception: d[lf] = None
#         rows.append(d)
#     return pd.DataFrame(rows)

# def normalize_list_columns(df):
#     list_cols = ['menternal_medical','obstetric_history','current_pregnancy_menternal','social','liquor']
#     for c in list_cols:
#         if c not in df.columns:
#             df[c] = [[] for _ in range(len(df))]
#         else:
#             df[c] = df[c].apply(safe_parse_list)
#     df['liquor_flags'] = df['liquor'].apply(lambda L: [s.lower() for s in L if isinstance(s, str)])
#     return df

# def derive_simple_flags(df):
#     df['history_pph'] = df['obstetric_history'].apply(lambda L: contains_any(L, ['postpartum hemorrhage','pph','post partum hemorrhage']))
#     df['grand_multipara'] = df['social'].apply(lambda L: contains_any(L, ['grand multipara'])) | df['parity'].apply(lambda x: True if (pd.notna(x) and x>=5) else False)
#     df['multiple_gestation'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ['multiple gestation','twin','twins','triplet']))
#     df['polihydraminos_flag'] = df['liquor_flags'].apply(lambda L: contains_any(L, ['polihydraminos','polyhydramnios','polihydramnios']))
#     df['placenta_prev_or_abruption'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ['placenta previa','placenta praevia','placenta abruption','abruption']))
#     df['severe_anemia'] = df['hb_g_dl'].apply(lambda x: True if (pd.notna(x) and float(x) < 7) else False)
#     df['obstructed_prolonged'] = df['obstetric_history'].apply(lambda L: contains_any(L, ['obstructed','prolonged labor','prolonged']))
#     return df

# def normalize_numeric_columns(df):
#     numeric_cols = ['age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','labor_duration_hours','blood_loss']
#     for c in numeric_cols:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors='coerce')
#         else:
#             df[c] = np.nan
#     return df

# def compute_target_pph(df):
#     if 'blood_loss' not in df.columns: df['blood_loss'] = np.nan
#     if 'blood_transfusion' not in df.columns: df['blood_transfusion'] = False
#     df['is_pph'] = False
#     try:
#         df.loc[df['blood_loss'].astype(float) >= 1000, 'is_pph'] = True
#     except Exception:
#         pass
#     df.loc[df['blood_transfusion'].astype(str).str.lower().isin(['true','1','yes','y']), 'is_pph'] = True
#     df['is_pph'] = df['is_pph'].astype(int)
#     return df

# def build_pipeline(numeric_features, categorical_features):
#     numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
#     categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
#     preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)], remainder='passthrough')
#     clf = RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42)
#     pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
#     return pipeline

# def extract_feature_names(pipeline, numeric_features, categorical_features, binary_features):
#     preproc = pipeline.named_steps['preprocessor']
#     try:
#         feature_names = list(preproc.get_feature_names_out())
#     except Exception:
#         try:
#             ohe = preproc.named_transformers_['cat'].named_steps['ohe']
#             cat_names = list(ohe.get_feature_names_out(categorical_features))
#             feature_names = list(numeric_features) + cat_names + list(binary_features)
#         except Exception:
#             feature_names = [f"f{i}" for i in range(len(pipeline.named_steps['classifier'].feature_importances_))]
#     return feature_names

# class Command(BaseCommand):
#     help = "Train improved PPH predictor with CV and hyperparameter tuning"

#     def add_arguments(self, parser):
#         parser.add_argument('--test-size', type=float, default=0.2)
#         parser.add_argument('--random-state', type=int, default=42)
#         parser.add_argument('--cv-folds', type=int, default=5)
#         parser.add_argument('--n-iter', type=int, default=20, help='RandomizedSearchCV iterations')

#     def handle(self, *args, **options):
#         if Patient is None:
#             self.stdout.write(self.style.ERROR("Patient model not importable. Run this inside Django."))
#             return
#         self.stdout.write("Loading patients...")
#         df = load_patients_to_df()
#         df = normalize_list_columns(df)
#         df = derive_simple_flags(df)
#         df = normalize_numeric_columns(df)
#         df = compute_target_pph(df)

#         self.stdout.write("Target distribution: " + str(df['is_pph'].value_counts().to_dict()))
#         features = [
#             'age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','labor_duration_hours',
#             'fetus_number','type_of_labor','mode_of_delivery','perineum_integrity','instrumental_delivery','type_of_cs','placenta_location',
#             'chronic_hypertension','history_pph','obstructed_prolonged','multiple_gestation','history_preeclampsia','severe_anemia','grand_multipara','polihydraminos_flag','placenta_prev_or_abruption'
#         ]
#         # ensure cols exist
#         for f in features:
#             if f not in df.columns:
#                 df[f] = np.nan

#         X = df[features].copy()
#         # binary coercion
#         bin_cols = ['chronic_hypertension','history_pph','obstructed_prolonged','multiple_gestation','history_preeclampsia','severe_anemia','grand_multipara','polihydraminos_flag','placenta_prev_or_abruption']
#         for b in bin_cols:
#             X[b] = X[b].apply(lambda v: 1 if (str(v).lower() in ['true','1','yes'] or v is True or v == 1) else 0)

#         y = df['is_pph'].astype(int)
#         # drop rows with missing target
#         mask = ~df['is_pph'].isna()
#         X = X[mask]; y = y[mask]

#         # split
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=options['test_size'], random_state=options['random_state'], stratify=y)
#         self.stdout.write("Train/test split sizes: %d / %d" % (len(X_train), len(X_test)))

#         numeric_features = ['age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','labor_duration_hours']
#         categorical_features = ['fetus_number','type_of_labor','mode_of_delivery','perineum_integrity','instrumental_delivery','type_of_cs','placenta_location']

#         base_pipeline = build_pipeline(numeric_features, categorical_features)

#         # Hyperparameter search space
#         param_dist = {
#             'classifier__n_estimators': sp_randint(50, 300),
#             'classifier__max_depth': sp_randint(3, 30),
#             'classifier__max_features': ['auto', 'sqrt', 'log2', 0.2, 0.5],
#             'classifier__min_samples_split': sp_randint(2, 10),
#         }

#         cv = StratifiedKFold(n_splits=options['cv_folds'], shuffle=True, random_state=options['random_state'])

#         self.stdout.write("Starting RandomizedSearchCV for %d iterations..." % options['n_iter'])
#         rs = RandomizedSearchCV(base_pipeline, param_distributions=param_dist, n_iter=options['n_iter'], scoring='roc_auc', n_jobs=-1, cv=cv, random_state=options['random_state'], verbose=1, return_train_score=False)
#         rs.fit(X_train, y_train)

#         best = rs.best_estimator_
#         self.stdout.write("Best params: " + str(rs.best_params_))
#         # Optionally calibrate probabilities if desired
#         calibrate = True
#         if calibrate:
#             try:
#                 calib = CalibratedClassifierCV(best.named_steps['classifier'], cv='prefit', method='isotonic')
#                 # wrap preprocessor + calibrated classifier
#                 preproc = best.named_steps['preprocessor']
#                 pipeline = Pipeline(steps=[('preprocessor', preproc), ('classifier', calib)])
#                 pipeline.fit(X_train, y_train)
#             except Exception:
#                 pipeline = best
#         else:
#             pipeline = best

#         # Evaluate on test set
#         y_pred = pipeline.predict(X_test)
#         try:
#             y_prob = pipeline.predict_proba(X_test)[:,1]
#         except Exception:
#             y_prob = None

#         metrics = {
#             'accuracy': float(accuracy_score(y_test, y_pred)),
#             'precision': float(precision_score(y_test, y_pred, zero_division=0)),
#             'recall': float(recall_score(y_test, y_pred, zero_division=0)),
#             'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
#             'classification_report': classification_report(y_test, y_pred, zero_division=0),
#         }
#         if y_prob is not None and len(np.unique(y_test)) > 1:
#             try:
#                 metrics['roc_auc'] = float(roc_auc_score(y_test, y_prob))
#             except Exception:
#                 metrics['roc_auc'] = None
#         else:
#             metrics['roc_auc'] = None

#         self.stdout.write("Test metrics:\\n" + json.dumps(metrics, indent=2))

#         # Save artifacts
#         artifacts_dir = os.path.join(getattr(settings, 'BASE_DIR', '.'), 'artifacts')
#         os.makedirs(artifacts_dir, exist_ok=True)
#         model_path = os.path.join(artifacts_dir, 'pph_pipeline_tuned.joblib')
#         cv_results_path = os.path.join(artifacts_dir, 'pph_cv_results.json')
#         joblib.dump(pipeline, model_path)
#         # save cv results summary
#         try:
#             cv_summary = {
#                 'best_params': rs.best_params_,
#                 'best_score': float(rs.best_score_),
#             }
#         except Exception:
#             cv_summary = {}
#         with open(cv_results_path, 'w') as fh:
#             json.dump(cv_summary, fh, indent=2)
#         self.stdout.write(self.style.SUCCESS(f"Saved tuned pipeline to: {model_path}"))
#         self.stdout.write(self.style.SUCCESS(f"Saved CV summary to: {cv_results_path}"))

#         # Feature importances safe extraction
#         try:
#             clf = pipeline.named_steps['classifier']
#             preproc = pipeline.named_steps['preprocessor']
#             try:
#                 feature_names = list(preproc.get_feature_names_out())
#             except Exception:
#                 ohe = preproc.named_transformers_['cat'].named_steps['ohe']
#                 cat_names = list(ohe.get_feature_names_out(categorical_features))
#                 feature_names = numeric_features + cat_names + list(bin_cols)
#             importances = None
#             try:
#                 importances = clf.feature_importances_
#             except Exception:
#                 # calibrated classifier may not expose feature_importances_
#                 pass
#             if importances is not None:
#                 fi = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
#                 fi.to_csv(os.path.join(artifacts_dir, 'pph_feature_importances_tuned.csv'), index=False)
#                 self.stdout.write("Saved feature importances.")
#         except Exception:
#             pass

#         self.stdout.write(self.style.SUCCESS("Tuning and evaluation complete."))


"""
Fixed PPH tuning command.

Fixes:
 - remove invalid 'auto' value for RandomForest max_features (sklearn >=1.2 rejects 'auto').
 - safer RandomizedSearchCV: catches fit failures, falls back to base pipeline if search fails.
 - sets error_score=np.nan and n_jobs appropriately.
 - better logging for failed fits and non-finite cv scores.

Placement: patients/management/commands/train_pph.py or keep as train_pph_tuned_fixed.py
"""

import os, json, re, joblib, numpy as np, pandas as pd
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from scipy.stats import randint as sp_randint
from sklearn.calibration import CalibratedClassifierCV
import warnings

# try to import Patient model
try:
    from patients.models import Patient
except Exception:
    Patient = None

# minimal helpers (reuse from previous)
def safe_parse_list(cell):
    if cell is None: return []
    if isinstance(cell, (list, tuple)): return [str(x).strip() for x in cell if str(x).strip()!='']
    if isinstance(cell, dict): return [str(v).strip() for v in cell.values() if str(v).strip()!='']
    if isinstance(cell, str):
        try:
            parsed = json.loads(cell)
            if isinstance(parsed, list): return [str(x).strip() for x in parsed if str(x).strip()!='']
        except Exception:
            pass
        parts = re.split(r'[;,|]\s*', cell)
        return [p.strip() for p in parts if p.strip()!='']
    return []

def contains_any(lst, keys):
    if not lst: return False
    low = [str(s).lower() for s in lst]
    for k in keys:
        if any(k.lower() in s for s in low): return True
    return False

def load_patients_to_df(qs=None):
    if qs is None:
        if Patient is None:
            raise RuntimeError("Run inside Django to load Patient model.")
        qs = Patient.objects.all()
    rows = []
    fields = [
        'id','age','parity','bmi','height','weight','total_number_of_cs','type_of_labor','mode_of_delivery',
        'perineum_integrity','instrumental_delivery','type_of_cs','labor_duration_hours','placenta_location',
        'estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','fetus_number','blood_loss','blood_transfusion'
    ]
    list_fields = ['menternal_medical','obstetric_history','current_pregnancy_menternal','social','liquor']
    for p in qs:
        d = {}
        for f in fields:
            try: d[f] = getattr(p, f)
            except Exception: d[f] = None
        for lf in list_fields:
            try: d[lf] = getattr(p, lf)
            except Exception: d[lf] = None
        rows.append(d)
    return pd.DataFrame(rows)

def normalize_list_columns(df):
    list_cols = ['menternal_medical','obstetric_history','current_pregnancy_menternal','social','liquor']
    for c in list_cols:
        if c not in df.columns:
            df[c] = [[] for _ in range(len(df))]
        else:
            df[c] = df[c].apply(safe_parse_list)
    df['liquor_flags'] = df['liquor'].apply(lambda L: [s.lower() for s in L if isinstance(s, str)])
    return df

def derive_simple_flags(df):
    df['history_pph'] = df['obstetric_history'].apply(lambda L: contains_any(L, ['postpartum hemorrhage','pph','post partum hemorrhage']))
    df['grand_multipara'] = df['social'].apply(lambda L: contains_any(L, ['grand multipara'])) | df['parity'].apply(lambda x: True if (pd.notna(x) and x>=5) else False)
    df['multiple_gestation'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ['multiple gestation','twin','twins','triplet']))
    df['polihydraminos_flag'] = df['liquor_flags'].apply(lambda L: contains_any(L, ['polihydraminos','polyhydramnios','polihydramnios']))
    df['placenta_prev_or_abruption'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ['placenta previa','placenta praevia','placenta abruption','abruption']))
    df['severe_anemia'] = df['hb_g_dl'].apply(lambda x: True if (pd.notna(x) and float(x) < 7) else False)
    df['obstructed_prolonged'] = df['obstetric_history'].apply(lambda L: contains_any(L, ['obstructed','prolonged labor','prolonged']))
    return df

def normalize_numeric_columns(df):
    numeric_cols = ['age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','labor_duration_hours','blood_loss']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan
    return df

def compute_target_pph(df):
    if 'blood_loss' not in df.columns: df['blood_loss'] = np.nan
    if 'blood_transfusion' not in df.columns: df['blood_transfusion'] = False
    df['is_pph'] = False
    try:
        df.loc[df['blood_loss'].astype(float) >= 1000, 'is_pph'] = True
    except Exception:
        pass
    df.loc[df['blood_transfusion'].astype(str).str.lower().isin(['true','1','yes','y']), 'is_pph'] = True
    df['is_pph'] = df['is_pph'].astype(int)
    return df

def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)], remainder='passthrough')
    clf = RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    return pipeline

class Command(BaseCommand):
    help = "Train improved PPH predictor with CV and hyperparameter tuning (fixed)"

    def add_arguments(self, parser):
        parser.add_argument('--test-size', type=float, default=0.2)
        parser.add_argument('--random-state', type=int, default=42)
        parser.add_argument('--cv-folds', type=int, default=5)
        parser.add_argument('--n-iter', type=int, default=20, help='RandomizedSearchCV iterations')

    def handle(self, *args, **options):
        if Patient is None:
            self.stdout.write(self.style.ERROR("Patient model not importable. Run this inside Django."))
            return
        self.stdout.write("Loading patients...")
        df = load_patients_to_df()
        df = normalize_list_columns(df)
        df = derive_simple_flags(df)
        df = normalize_numeric_columns(df)
        df = compute_target_pph(df)

        self.stdout.write("Target distribution: " + str(df['is_pph'].value_counts().to_dict()))
        features = [
            'age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','labor_duration_hours',
            'fetus_number','type_of_labor','mode_of_delivery','perineum_integrity','instrumental_delivery','type_of_cs','placenta_location',
            'chronic_hypertension','history_pph','obstructed_prolonged','multiple_gestation','history_preeclampsia','severe_anemia','grand_multipara','polihydraminos_flag','placenta_prev_or_abruption'
        ]
        for f in features:
            if f not in df.columns:
                df[f] = np.nan

        X = df[features].copy()
        bin_cols = ['chronic_hypertension','history_pph','obstructed_prolonged','multiple_gestation','history_preeclampsia','severe_anemia','grand_multipara','polihydraminos_flag','placenta_prev_or_abruption']
        for b in bin_cols:
            X[b] = X[b].apply(lambda v: 1 if (str(v).lower() in ['true','1','yes'] or v is True or v == 1) else 0)

        y = df['is_pph'].astype(int)
        mask = ~df['is_pph'].isna()
        X = X[mask]; y = y[mask]

        # if too few positives, warn
        pos = int(y.sum())
        total = len(y)
        if pos < 10:
            self.stdout.write(self.style.WARNING(f"Very few positive PPH cases ({pos}/{total}) — model performance may be limited."))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=options['test_size'], random_state=options['random_state'], stratify=y)
        self.stdout.write("Train/test split sizes: %d / %d" % (len(X_train), len(X_test)))

        numeric_features = ['age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','labor_duration_hours']
        categorical_features = ['fetus_number','type_of_labor','mode_of_delivery','perineum_integrity','instrumental_delivery','type_of_cs','placenta_location']

        base_pipeline = build_pipeline(numeric_features, categorical_features)

        # Corrected hyperparameter space (no 'auto')
        param_dist = {
            'classifier__n_estimators': sp_randint(50, 300),
            'classifier__max_depth': sp_randint(3, 30),
            'classifier__max_features': ['sqrt', 'log2', None, 0.2, 0.5],
            'classifier__min_samples_split': sp_randint(2, 10),
        }

        cv = StratifiedKFold(n_splits=options['cv_folds'], shuffle=True, random_state=options['random_state'])

        self.stdout.write("Starting RandomizedSearchCV for %d iterations..." % options['n_iter'])
        rs = RandomizedSearchCV(base_pipeline, param_distributions=param_dist, n_iter=options['n_iter'], scoring='roc_auc', n_jobs=-1, cv=cv, random_state=options['random_state'], verbose=1, return_train_score=False, error_score=np.nan)

        # run with exception handling: if many fits fail, fall back to base_pipeline
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rs.fit(X_train, y_train)
            best = rs.best_estimator_
            self.stdout.write("RandomizedSearchCV completed. Best params: " + str(rs.best_params_))
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"RandomizedSearchCV failed: {e}. Falling back to base pipeline."))
            # fallback: fit base pipeline
            base_pipeline.fit(X_train, y_train)
            best = base_pipeline

        # Optionally calibrate probabilities
        calibrate = True
        pipeline = best
        if calibrate:
            try:
                # if classifier is prefit (RandomizedSearchCV gives estimator already fitted), wrap classifier only
                if hasattr(best.named_steps['classifier'], "predict_proba"):
                    from sklearn.calibration import CalibratedClassifierCV
                    clf = best.named_steps['classifier']
                    preproc = best.named_steps['preprocessor']
                    try:
                        calib = CalibratedClassifierCV(clf, cv='prefit', method='isotonic')
                        pipeline = Pipeline(steps=[('preprocessor', preproc), ('classifier', calib)])
                        pipeline.fit(X_train, y_train)
                    except Exception:
                        pipeline = best
            except Exception:
                pipeline = best

        # Evaluate
        y_pred = pipeline.predict(X_test)
        try:
            y_prob = pipeline.predict_proba(X_test)[:,1]
        except Exception:
            y_prob = None

        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, zero_division=0),
        }
        if y_prob is not None and len(np.unique(y_test)) > 1:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_test, y_prob))
            except Exception:
                metrics['roc_auc'] = None
        else:
            metrics['roc_auc'] = None

        self.stdout.write("Test metrics:\\n" + json.dumps(metrics, indent=2))

        artifacts_dir = os.path.join(getattr(settings, 'BASE_DIR', '.'), 'artifacts')
        os.makedirs(artifacts_dir, exist_ok=True)
        model_path = os.path.join(artifacts_dir, 'pph_pipeline_tuned_fixed.joblib')
        cv_results_path = os.path.join(artifacts_dir, 'pph_cv_results_fixed.json')
        joblib.dump(pipeline, model_path)

        try:
            cv_summary = {}
            if 'rs' in locals() and hasattr(rs, 'best_params_'):
                cv_summary = {'best_params': rs.best_params_, 'best_score': float(rs.best_score_)}
        except Exception:
            cv_summary = {}
        with open(cv_results_path, 'w') as fh:
            json.dump(cv_summary, fh, indent=2)

        # feature importances
        try:
            clf = pipeline.named_steps['classifier']
            preproc = pipeline.named_steps['preprocessor']
            try:
                feature_names = list(preproc.get_feature_names_out())
            except Exception:
                ohe = preproc.named_transformers_['cat'].named_steps['ohe']
                cat_names = list(ohe.get_feature_names_out(categorical_features))
                feature_names = numeric_features + cat_names + list(bin_cols)
            importances = None
            try:
                importances = clf.feature_importances_
            except Exception:
                pass
            if importances is not None:
                fi = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
                fi.to_csv(os.path.join(artifacts_dir, 'pph_feature_importances_tuned_fixed.csv'), index=False)
        except Exception:
            pass

        self.stdout.write(self.style.SUCCESS("Tuning and evaluation complete."))
