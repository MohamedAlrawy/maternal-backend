# import json
# import re
# import os
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
# from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report

# from patients.models import Patient  # ensure app name and model import path

# # ------------------- helper functions -------------------
# def safe_parse_list(cell):
#     if cell is None:
#         return []
#     if isinstance(cell, (list, tuple)):
#         return [str(x).strip() for x in cell if str(x).strip() != ""]
#     try:
#         parsed = json.loads(cell)
#         if isinstance(parsed, list):
#             return [str(x).strip() for x in parsed]
#     except Exception:
#         pass
#     if isinstance(cell, str):
#         parts = re.split(r'[;,|]\s*', cell)
#         return [p.strip() for p in parts if p.strip() != ""]
#     return []

# def contains_any(lst, keys):
#     if not lst: 
#         return False
#     low = [s.lower() for s in lst]
#     for k in keys:
#         if any(k.lower() in s for s in low):
#             return True
#     return False

# # deterministic rules (prob% and reason)
# RULES = [
#     (lambda r: (r.get('total_number_of_cs', 0) == 1), 35, "Previous 1 CS -> ~30-40% repeat CS"),
#     (lambda r: (r.get('total_number_of_cs', 0) == 2), 90, "Two prior CS -> ≈90% repeat CS"),
#     (lambda r: (r.get('total_number_of_cs', 0) >= 3), 99, "Three or more CS -> ≈100% CS"),
#     (lambda r: r.get('has_placenta_previa', False), 99, "Placenta previa -> ≈100% CS"),
#     (lambda r: r.get('has_placenta_abruption', False), 85, "Placenta abruption -> 80-90% CS"),
#     (lambda r: r.get('non_cephalic', False), 90, "Non-cephalic presentation -> 85-95% CS"),
#     (lambda r: r.get('multiple_gestation', False), 60, "Multiple gestation -> 50-70% CS"),
#     (lambda r: (r.get('estimated_fetal_weight_by_gm', 0) >= 4000), 50, "Estimated fetal weight ≥4000g -> 40-60% CS"),
#     (lambda r: r.get('chronic_hypertension', False), 50, "Chronic hypertension -> 40-60% CS"),
#     (lambda r: r.get('diabetes_any', False), 45, "Diabetes -> 35-55% CS"),
#     (lambda r: (('pre_eclampsia_eclampsia_hellp' in str(r.get('cs_indication', '')).lower()) or r.get('history_preeclampsia', False)), 60, "Preeclampsia/HELLP -> 50-70% CS"),
#     (lambda r: (not pd.isna(r.get('hb_g_dl')) and float(r.get('hb_g_dl', 999)) < 7), 25, "Severe anemia (Hb<7) -> 20-30% CS"),
#     (lambda r: contains_any(r.get('menternal_medical', []), ["cardiac disease", "cardiac"]), 45, "Cardiac disease -> 30-60% CS"),
#     (lambda r: contains_any(r.get('menternal_medical', []), ["hiv", "immunocompromised"]), 20, "HIV/immunocompromised -> 15-25%"),
#     (lambda r: contains_any(r.get('obstetric_history', []), ["uterine rupture"]), 99, "Prior uterine rupture -> ≈100% CS"),
#     (lambda r: (str(r.get('ctg_category','')).lower().find('category_ii')>=0 or str(r.get('ctg_category','')).lower().find('category ii')>=0), 70, "CTG Category II -> 70% CS"),
#     (lambda r: (str(r.get('ctg_category','')).lower().find('category_iii')>=0 or str(r.get('ctg_category','')).lower().find('category iii')>=0), 95, "CTG Category III -> 90-100% CS"),
#     (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ["ivf", "icsi", "pregnancy post ivf", "pregnancy post icsi"]), 85, "Post IVF/ICSI -> 80-90% CS"),
# ]

# def deterministic_check(rowdict):
#     matches = []
#     for cond, prob, reason in RULES:
#         try:
#             if cond(rowdict):
#                 matches.append((prob, reason))
#         except Exception:
#             continue
#     if not matches:
#         return False, None, None
#     best = max(matches, key=lambda x: x[0])
#     return True, best[0], best[1]

# # ------------------- Command -------------------
# class Command(BaseCommand):
#     help = "Train CS (cesarean) predictor from Patient table and save pipeline"

#     def handle(self, *args, **options):
#         qs = Patient.objects.all()
#         n_patients = qs.count()
#         if qs.count() == 0:
#             self.stdout.write(self.style.ERROR("No Patient rows found."))
#             return

#         # Convert queryset to DataFrame
#         rows = []
#         for p in qs:
#             d = {}
#             # pull simple fields
#             for f in ['id','name','file_number','patient_id','age','parity','bmi','bp','Hb_g_dL','hb_g_dl',
#                       'total_number_of_cs','presentation','fetus_number','cervical_dilatation_at_admission',
#                       'ctg_category','estimated_fetal_weight_by_gm','rupture_duration_hour','mode_of_delivery','cs_indication']:
#                 d[f] = getattr(p, f, None)
#             # parse json/list fields - ensure consistent keys present
#             for jsoncol in ['menternal_medical','obstetric_history','current_pregnancy_menternal','current_pregnancy_fetal','social']:
#                 d[jsoncol] = getattr(p, jsoncol, None)
#             rows.append(d)

#         df = pd.DataFrame(rows)

#         # normalize list columns
#         list_cols = ['menternal_medical','obstetric_history','current_pregnancy_menternal','current_pregnancy_fetal','social']
#         for c in list_cols:
#             df[c] = df[c].apply(safe_parse_list)

#         # compute binary flags used by rules & features
#         df['has_placenta_previa'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ["placenta previa", "placenta praevia", "placenta previa/abruption"]))
#         df['has_placenta_abruption'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ["placenta abruption", "abruption"]))
#         df['non_cephalic'] = df['current_pregnancy_fetal'].apply(lambda L: contains_any(L, ["breech", "non-cephalic", "transverse", "oblique"]))
#         df['multiple_gestation'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ["multiple gestation", "twin", "twins", "triplet"]))
#         df['chronic_hypertension'] = df['menternal_medical'].apply(lambda L: contains_any(L, ["chronic hypertension", "hypertension"]))
#         df['diabetes_any'] = df['menternal_medical'].apply(lambda L: contains_any(L, ["diabetes", "gdm", "gestational diabetes"]))
#         df['gdm'] = df['current_pregnancy_menternal'].apply(lambda L: contains_any(L, ["gdm", "gestational diabetes"]))
#         df['history_preeclampsia'] = df['obstetric_history'].apply(lambda L: contains_any(L, ["preeclampsia", "pre-eclampsia", "pre eclampsia"]))

#         # unify hb column
#         if 'Hb_g_dL' in df.columns and df['Hb_g_dL'].notna().any():
#             df['hb_g_dl'] = pd.to_numeric(df['Hb_g_dL'], errors='coerce')
#         else:
#             df['hb_g_dl'] = pd.to_numeric(df.get('hb_g_dl', np.nan), errors='coerce')
#         df['severe_anemia'] = df['hb_g_dl'].apply(lambda x: True if (not pd.isna(x) and float(x) < 7) else False)

#         # normalize total_number_of_cs numeric
#         if 'total_number_of_cs' in df.columns:
#             df['total_number_of_cs'] = pd.to_numeric(df['total_number_of_cs'], errors='coerce').fillna(0)
#         else:
#             df['total_number_of_cs'] = 0

#         # normalize mode_of_delivery -> target
#         df['mode_of_delivery_norm'] = df['mode_of_delivery'].astype(str).str.lower()
#         df['is_cs'] = df['mode_of_delivery_norm'].apply(lambda s: 1 if ('c' in s and ('s' in s or 's' in s)) or 'cesarean' in s or 'cs' in s else (1 if 'c.s' in s else 0))
#         df['is_cs'] = df['is_cs'].astype(int)

#         # apply deterministic rules
#         df['deterministic_override'] = False
#         df['deterministic_prob'] = np.nan
#         df['deterministic_reason'] = None
#         for idx, row in df.iterrows():
#             rowdict = row.to_dict()
#             override, prob, reason = deterministic_check(rowdict)
#             df.at[idx, 'deterministic_override'] = override
#             df.at[idx, 'deterministic_prob'] = prob
#             df.at[idx, 'deterministic_reason'] = reason

#         # exclude extremely-high-confidence rule rows from training (optional)
#         TRAIN_EXCLUDE_HIGH_CONF = 80
#         train_mask = ~( (df['deterministic_override']) & (df['deterministic_prob'] >= TRAIN_EXCLUDE_HIGH_CONF) )
#         train_df = df[train_mask].copy()
#         if len(train_df) < 10:
#             self.stdout.write(self.style.WARNING("Very few rows to train on after excluding high-confidence rule rows. Training anyway."))

#         # features to use (same as previously specified)
#         features = [
#             'age','parity','bmi','chronic_hypertension','diabetes_any','total_number_of_cs',
#             'non_cephalic','multiple_gestation','gdm','history_preeclampsia','severe_anemia',
#             'presentation','fetus_number','cervical_dilatation_at_admission','ctg_category',
#             'estimated_fetal_weight_by_gm','rupture_duration_hour'
#         ]
#         for f in features:
#             if f not in train_df.columns:
#                 train_df[f] = np.nan

#         numeric_features = ['age','parity','bmi','total_number_of_cs','cervical_dilatation_at_admission','estimated_fetal_weight_by_gm']
#         categorical_features = ['presentation','fetus_number','ctg_category','rupture_duration_hour']
#         binary_features = ['chronic_hypertension','diabetes_any','non_cephalic','multiple_gestation','gdm','history_preeclampsia','severe_anemia']

#         X = train_df[numeric_features + categorical_features + binary_features].copy()
#         # binarize booleans
#         for b in binary_features:
#             X[b] = X[b].apply(lambda v: 1 if (str(v).lower() in ['true','1','yes'] or v is True) else 0)

#         y = train_df['is_cs'].astype(int)

#         # Build pipeline
#         numeric_transformer = Pipeline(steps=[
#             ('imputer', SimpleImputer(strategy='median')),
#             ('scaler', StandardScaler()),
#         ])
#         categorical_transformer = Pipeline(steps=[
#             ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#             ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
#         ])
#         preprocessor = ColumnTransformer(transformers=[
#             ('num', numeric_transformer, numeric_features),
#             ('cat', categorical_transformer, categorical_features),
#         ], remainder='passthrough')

#         clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
#         pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])

#         # train/test split
#         try:
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#         except Exception:
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         pipeline.fit(X_train, y_train)

#         y_pred = pipeline.predict(X_test)
#         y_prob = pipeline.predict_proba(X_test)[:,1] if hasattr(pipeline, "predict_proba") else None
#         self.stdout.write("\nTraining Report")
#         self.stdout.write(" Number of patient records used: {}".format(n_patients))
#         self.stdout.write("Model evaluation:")
#         self.stdout.write(" Accuracy: {:.3f}".format(accuracy_score(y_test, y_pred)))
#         self.stdout.write(" Precision: {:.3f}".format(precision_score(y_test, y_pred, zero_division=0)))
#         self.stdout.write(" Recall: {:.3f}".format(recall_score(y_test, y_pred, zero_division=0)))
#         if y_prob is not None:
#             self.stdout.write(" ROC-AUC: {:.3f}".format(roc_auc_score(y_test, y_prob)))

#         self.stdout.write("Classification report:\n" + classification_report(y_test, y_pred))

#         # save pipeline
#         artifacts_dir = os.path.join(settings.BASE_DIR, 'artifacts')
#         os.makedirs(artifacts_dir, exist_ok=True)
#         model_path = os.path.join(artifacts_dir, 'cs_pipeline.joblib')
#         joblib.dump(pipeline, model_path)
#         self.stdout.write(self.style.SUCCESS(f"Saved pipeline to {model_path}"))

#         # save feature importance (global) for quick inspection
#         try:
#             preproc = pipeline.named_steps['preprocessor']
#             ohe = preproc.named_transformers_['cat'].named_steps['ohe']
#             cat_names = ohe.get_feature_names_out(categorical_features)
#             feature_names = list(numeric_features) + list(cat_names) + list(binary_features)
#             importances = pipeline.named_steps['classifier'].feature_importances_
#             feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
#             self.stdout.write("Top feature importances:")
#             for name, imp in feat_imp[:10]:
#                 self.stdout.write(f" - {name}: {imp:.4f}")
#         except Exception:
#             pass

#         self.stdout.write(self.style.SUCCESS("Training complete."))

"""
Refactored Cesarean Section (CS) Predictor Training Module

A modular, maintainable training pipeline for predicting cesarean deliveries
using clinical data with deterministic rules and machine learning.

Architecture:
- Data Layer: Patient data extraction and parsing
- Feature Engineering: Clinical flag derivation and normalization
- Rule Engine: Deterministic clinical decision rules
- ML Pipeline: Sklearn pipeline with preprocessing and classification
- Persistence: Model and metrics serialization

Usage:
    python manage.py train_cs --exclude-high-confidence 80 --test-size 0.2
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
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
    accuracy_score, precision_score, recall_score, 
    roc_auc_score, classification_report, confusion_matrix
)

try:
    from patients.models import Patient
except ImportError:
    Patient = None

# Configure logging
logger = logging.getLogger(__name__)


# ==================== Configuration ====================

@dataclass
class FeatureConfig:
    """Feature configuration for the ML pipeline"""
    numeric: List[str]
    categorical: List[str]
    binary: List[str]
    list_columns: List[str]
    
    @property
    def all_features(self) -> List[str]:
        return self.numeric + self.categorical + self.binary


# Default feature configuration
DEFAULT_FEATURES = FeatureConfig(
    numeric=[
        'age', 'parity', 'bmi', 'total_number_of_cs',
        'cervical_dilatation_at_admission', 'estimated_fetal_weight_by_gm'
    ],
    categorical=[
        'presentation', 'fetus_number', 'ctg_category', 'rupture_duration_hour'
    ],
    binary=[
        'chronic_hypertension', 'diabetes_any', 'non_cephalic',
        'multiple_gestation', 'gdm', 'history_preeclampsia', 'severe_anemia'
    ],
    list_columns=[
        'menternal_medical', 'obstetric_history',
        'current_pregnancy_menternal', 'current_pregnancy_fetal', 'social'
    ]
)


@dataclass
class ClinicalRule:
    """Represents a deterministic clinical decision rule"""
    condition: Callable[[Dict], bool]
    probability: float
    reason: str
    
    def evaluate(self, patient_data: Dict) -> Tuple[bool, Optional[float], Optional[str]]:
        """Safely evaluate rule and return (matched, probability, reason)"""
        try:
            if self.condition(patient_data):
                return True, self.probability, self.reason
        except Exception as e:
            logger.debug(f"Rule evaluation failed: {e}")
        return False, None, None


# ==================== Data Parsing Utilities ====================

class DataParser:
    """Utilities for parsing heterogeneous clinical data formats"""
    
    @staticmethod
    def parse_list(cell: Any) -> List[str]:
        """
        Parse various list representations into standardized list[str]
        
        Handles: Python lists/tuples, JSON strings, delimited strings, dicts
        """
        if cell is None:
            return []
        
        # Native Python collections
        if isinstance(cell, (list, tuple)):
            return [str(x).strip() for x in cell if str(x).strip()]
        
        # Dictionary (extract values)
        if isinstance(cell, dict):
            return [str(v).strip() for v in cell.values() if str(v).strip()]
        
        # String representations
        if isinstance(cell, str):
            # Try JSON parsing first
            try:
                parsed = json.loads(cell)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Fall back to delimiter splitting
            parts = re.split(r'[;,|]\s*', cell)
            return [p.strip() for p in parts if p.strip()]
        
        return []
    
    @staticmethod
    def contains_any(items: List[str], keywords: List[str]) -> bool:
        """
        Check if any keyword appears in any item (case-insensitive substring match)
        
        Used for fuzzy matching clinical terms in free-text fields
        """
        if not items:
            return False
        
        items_lower = [str(s).lower() for s in items]
        return any(
            keyword.lower() in item
            for keyword in keywords
            for item in items_lower
        )


# ==================== Clinical Rules Engine ====================

class ClinicalRulesEngine:
    """Manages deterministic clinical decision rules for CS prediction"""
    
    def __init__(self):
        self.rules = self._define_rules()
    
    def _define_rules(self) -> List[ClinicalRule]:
        """Define all clinical rules with evidence-based probabilities"""
        return [
            # Prior CS history
            ClinicalRule(
                lambda r: r.get('total_number_of_cs', 0) == 1,
                35,
                "Previous 1 CS → 30-40% repeat CS risk"
            ),
            ClinicalRule(
                lambda r: r.get('total_number_of_cs', 0) == 2,
                90,
                "Two prior CS → ≈90% repeat CS risk"
            ),
            ClinicalRule(
                lambda r: r.get('total_number_of_cs', 0) >= 3,
                99,
                "Three or more CS → ≈100% repeat CS"
            ),
            
            # Placental complications
            ClinicalRule(
                lambda r: r.get('has_placenta_previa', False),
                99,
                "Placenta previa → ≈100% CS indication"
            ),
            ClinicalRule(
                lambda r: r.get('has_placenta_abruption', False),
                85,
                "Placenta abruption → 80-90% CS"
            ),
            
            # Fetal presentation and gestation
            ClinicalRule(
                lambda r: r.get('non_cephalic', False),
                90,
                "Non-cephalic presentation → 85-95% CS"
            ),
            ClinicalRule(
                lambda r: r.get('multiple_gestation', False),
                60,
                "Multiple gestation → 50-70% CS"
            ),
            ClinicalRule(
                lambda r: r.get('estimated_fetal_weight_by_gm', 0) >= 4000,
                50,
                "Macrosomia (≥4000g) → 40-60% CS"
            ),
            
            # Maternal medical conditions
            ClinicalRule(
                lambda r: r.get('chronic_hypertension', False),
                50,
                "Chronic hypertension → 40-60% CS"
            ),
            ClinicalRule(
                lambda r: r.get('diabetes_any', False),
                45,
                "Diabetes → 35-55% CS"
            ),
            ClinicalRule(
                lambda r: (
                    'pre_eclampsia_eclampsia_hellp' in str(r.get('cs_indication', '')).lower()
                    or r.get('history_preeclampsia', False)
                ),
                60,
                "Preeclampsia/HELLP → 50-70% CS"
            ),
            ClinicalRule(
                lambda r: not pd.isna(r.get('hb_g_dl')) and float(r.get('hb_g_dl', 999)) < 7,
                25,
                "Severe anemia (Hb<7) → 20-30% CS"
            ),
            ClinicalRule(
                lambda r: DataParser.contains_any(
                    r.get('menternal_medical', []),
                    ["cardiac disease", "cardiac"]
                ),
                45,
                "Cardiac disease → 30-60% CS"
            ),
            ClinicalRule(
                lambda r: DataParser.contains_any(
                    r.get('menternal_medical', []),
                    ["hiv", "immunocompromised"]
                ),
                20,
                "HIV/immunocompromised → 15-25% CS"
            ),
            
            # Obstetric history
            ClinicalRule(
                lambda r: DataParser.contains_any(
                    r.get('obstetric_history', []),
                    ["uterine rupture"]
                ),
                99,
                "Prior uterine rupture → ≈100% CS"
            ),
            
            # Fetal monitoring
            ClinicalRule(
                lambda r: (
                    'category_ii' in str(r.get('ctg_category', '')).lower().replace(' ', '_')
                    or 'category ii' in str(r.get('ctg_category', '')).lower()
                ),
                70,
                "CTG Category II → 70% CS"
            ),
            ClinicalRule(
                lambda r: (
                    'category_iii' in str(r.get('ctg_category', '')).lower().replace(' ', '_')
                    or 'category iii' in str(r.get('ctg_category', '')).lower()
                ),
                95,
                "CTG Category III → 90-100% CS"
            ),
            
            # Assisted reproduction
            ClinicalRule(
                lambda r: DataParser.contains_any(
                    r.get('current_pregnancy_menternal', []),
                    ["ivf", "icsi", "pregnancy post ivf", "pregnancy post icsi"]
                ),
                85,
                "Post IVF/ICSI → 80-90% CS"
            ),
        ]
    
    def evaluate_patient(self, patient_data: Dict) -> Tuple[bool, Optional[float], Optional[str]]:
        """
        Evaluate all rules for a patient and return highest-priority match
        
        Returns:
            (has_override, probability, reason)
        """
        matches = []
        
        for rule in self.rules:
            matched, prob, reason = rule.evaluate(patient_data)
            if matched:
                matches.append((prob, reason))
        
        if not matches:
            return False, None, None
        
        # Return highest-probability match
        best_match = max(matches, key=lambda x: x[0])
        return True, best_match[0], best_match[1]


# ==================== Data Loading and Processing ====================

class PatientDataLoader:
    """Handles loading and initial processing of patient data"""
    
    REQUIRED_FIELDS = [
        'id', 'name', 'file_number', 'patient_id', 'age', 'parity', 'bmi',
        'bp', 'Hb_g_dL', 'hb_g_dl', 'total_number_of_cs', 'presentation',
        'fetus_number', 'cervical_dilatation_at_admission', 'ctg_category',
        'estimated_fetal_weight_by_gm', 'rupture_duration_hour',
        'mode_of_delivery', 'cs_indication', 'menternal_medical',
        'obstetric_history', 'current_pregnancy_menternal',
        'current_pregnancy_fetal', 'social'
    ]
    
    @staticmethod
    def load_from_queryset(queryset=None) -> pd.DataFrame:
        """Load patient data from Django queryset into DataFrame"""
        if queryset is None:
            if Patient is None:
                raise RuntimeError("Patient model unavailable. Run within Django project.")
            queryset = Patient.objects.all()
        
        rows = []
        for patient in queryset:
            row = {}
            for field in PatientDataLoader.REQUIRED_FIELDS:
                try:
                    row[field] = getattr(patient, field, None)
                except AttributeError:
                    row[field] = None
            rows.append(row)
        
        return pd.DataFrame(rows)


class FeatureEngineering:
    """Handles feature derivation and data normalization"""
    
    def __init__(self, feature_config: FeatureConfig):
        self.config = feature_config
        self.parser = DataParser()
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        df = df.copy()
        df = self._normalize_list_columns(df)
        df = self._derive_clinical_flags(df)
        df = self._normalize_hemoglobin(df)
        df = self._normalize_numeric_columns(df)
        df = self._compute_target(df)
        return df
    
    def _normalize_list_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert list-like columns to standardized list format"""
        for col in self.config.list_columns:
            if col not in df.columns:
                df[col] = [[] for _ in range(len(df))]
            else:
                df[col] = df[col].apply(self.parser.parse_list)
        return df
    
    def _derive_clinical_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive binary clinical flags from list columns"""
        flag_definitions = {
            'has_placenta_previa': (
                'current_pregnancy_menternal',
                ["placenta previa", "placenta praevia", "placenta previa/abruption"]
            ),
            'has_placenta_abruption': (
                'current_pregnancy_menternal',
                ["placenta abruption", "abruption"]
            ),
            'non_cephalic': (
                'current_pregnancy_fetal',
                ["breech", "non-cephalic", "transverse", "oblique"]
            ),
            'multiple_gestation': (
                'current_pregnancy_menternal',
                ["multiple gestation", "twin", "twins", "triplet"]
            ),
            'chronic_hypertension': (
                'menternal_medical',
                ["chronic hypertension", "hypertension"]
            ),
            'diabetes_any': (
                'menternal_medical',
                ["diabetes", "gdm", "gestational diabetes"]
            ),
            'gdm': (
                'current_pregnancy_menternal',
                ["gdm", "gestational diabetes"]
            ),
            'history_preeclampsia': (
                'obstetric_history',
                ["preeclampsia", "pre-eclampsia", "pre eclampsia"]
            ),
        }
        
        for flag_name, (source_col, keywords) in flag_definitions.items():
            df[flag_name] = df[source_col].apply(
                lambda items: self.parser.contains_any(items, keywords)
            )
        
        return df
    
    def _normalize_hemoglobin(self, df: pd.DataFrame) -> pd.DataFrame:
        """Unify hemoglobin columns and derive severe anemia flag"""
        # Prefer Hb_g_dL if available, otherwise use hb_g_dl
        if 'Hb_g_dL' in df.columns and df['Hb_g_dL'].notna().any():
            df['hb_g_dl'] = pd.to_numeric(df['Hb_g_dL'], errors='coerce')
        else:
            if 'hb_g_dl' not in df.columns:
                df['hb_g_dl'] = np.nan
            else:
                df['hb_g_dl'] = pd.to_numeric(df['hb_g_dl'], errors='coerce')
        
        df['severe_anemia'] = df['hb_g_dl'].apply(
            lambda x: not pd.isna(x) and float(x) < 7
        )
        
        return df
    
    def _normalize_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert numeric columns to proper dtype with sensible defaults"""
        # CS count (default to 0 if missing)
        if 'total_number_of_cs' in df.columns:
            df['total_number_of_cs'] = pd.to_numeric(
                df['total_number_of_cs'], errors='coerce'
            ).fillna(0)
        else:
            df['total_number_of_cs'] = 0
        
        # Other numeric features
        for col in self.config.numeric:
            if col == 'total_number_of_cs':
                continue  # Already handled
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = np.nan
        
        return df
    
    def _compute_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive binary target variable from mode_of_delivery"""
        if 'mode_of_delivery' not in df.columns:
            df['mode_of_delivery'] = None
        
        df['mode_of_delivery_norm'] = (
            df['mode_of_delivery']
            .fillna('')
            .astype(str)
            .str.lower()
        )
        
        # Detect CS keywords
        cs_keywords = ['cesarean', 'cs', 'c/s', 'c.s']
        df['is_cs'] = df['mode_of_delivery_norm'].apply(
            lambda s: int(any(kw in s for kw in cs_keywords))
        )
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Prepare feature matrix X with missing value summary
        
        Returns:
            (X, missing_summary)
        """
        X = df.copy()
        missing_summary = {}
        
        # Ensure all features exist
        for feature in self.config.all_features:
            if feature not in X.columns:
                X[feature] = np.nan
            missing_summary[feature] = int(X[feature].isna().sum())
        
        # Binarize boolean features
        for binary_feature in self.config.binary:
            X[binary_feature] = X[binary_feature].apply(
                lambda v: int(str(v).lower() in ['true', '1', 'yes'] or v is True or v == 1)
            )
        
        return X[self.config.all_features], missing_summary


# ==================== ML Pipeline ====================

class CSPredictorPipeline:
    """Manages the machine learning pipeline for CS prediction"""
    
    def __init__(self, feature_config: FeatureConfig):
        self.config = feature_config
        self.pipeline = None
    
    def build(self) -> Pipeline:
        """Construct the sklearn pipeline"""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.config.numeric),
                ('cat', categorical_transformer, self.config.categorical)
            ],
            remainder='passthrough'
        )
        
        classifier = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        
        return self.pipeline
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train the pipeline"""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not built. Call build() first.")
        self.pipeline.fit(X_train, y_train)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model and return metrics dictionary"""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not trained.")
        
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }
        
        # ROC-AUC (only if both classes present)
        if len(np.unique(y_test)) > 1:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_test, y_prob))
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
                metrics['roc_auc'] = None
        
        return metrics
    
    def get_feature_importances(self) -> pd.DataFrame:
        """Extract feature importances with names"""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not trained.")
        
        try:
            # Get feature names after preprocessing
            preprocessor = self.pipeline.named_steps['preprocessor']
            ohe = preprocessor.named_transformers_['cat'].named_steps['ohe']
            cat_names = ohe.get_feature_names_out(self.config.categorical)
            
            all_names = (
                list(self.config.numeric) +
                list(cat_names) +
                list(self.config.binary)
            )
            
            # Get importances from classifier
            importances = self.pipeline.named_steps['classifier'].feature_importances_
            
            return pd.DataFrame({
                'feature': all_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
        except Exception as e:
            logger.error(f"Could not extract feature importances: {e}")
            return pd.DataFrame()


# ==================== Artifacts Manager ====================

class ArtifactsManager:
    """Handles saving and loading of model artifacts"""
    
    def __init__(self, artifacts_dir: Optional[Path] = None):
        if artifacts_dir is None:
            artifacts_dir = Path(settings.BASE_DIR) / 'artifacts'
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    def save_pipeline(self, pipeline: Pipeline, filename: str = 'cs_pipeline.joblib') -> Path:
        """Save sklearn pipeline"""
        path = self.artifacts_dir / filename
        joblib.dump(pipeline, path)
        return path
    
    def save_metrics(self, metrics: Dict, filename: str = 'cs_metrics.json') -> Path:
        """Save metrics dictionary as JSON"""
        path = self.artifacts_dir / filename
        
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16,
                                np.int32, np.int64, np.uint8, np.uint16,
                                np.uint32, np.uint64)):
                return int(obj)
            if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            return obj
        
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2, default=convert)
        
        return path
    
    def save_feature_importances(self, 
                                   importances_df: pd.DataFrame,
                                   filename: str = 'cs_feature_importances.csv') -> Path:
        """Save feature importances as CSV"""
        path = self.artifacts_dir / filename
        importances_df.to_csv(path, index=False)
        return path
    
    def load_pipeline(self, filename: str = 'cs_pipeline.joblib') -> Pipeline:
        """Load saved pipeline"""
        path = self.artifacts_dir / filename
        return joblib.load(path)


# ==================== Django Management Command ====================

class Command(BaseCommand):
    help = 'Train CS predictor with deterministic rules and ML pipeline'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--exclude-high-confidence',
            type=int,
            default=80,
            help='Exclude deterministic rule matches >= this probability from training'
        )
        parser.add_argument(
            '--test-size',
            type=float,
            default=0.2,
            help='Fraction of data to use for testing'
        )
        parser.add_argument(
            '--random-state',
            type=int,
            default=42,
            help='Random seed for reproducibility'
        )
    
    def handle(self, *args, **options):
        """Main training workflow"""
        self.stdout.write(self.style.SUCCESS('=== CS Predictor Training ===\n'))
        
        # Initialize components
        loader = PatientDataLoader()
        engineer = FeatureEngineering(DEFAULT_FEATURES)
        rules_engine = ClinicalRulesEngine()
        ml_pipeline = CSPredictorPipeline(DEFAULT_FEATURES)
        artifacts_mgr = ArtifactsManager()
        
        # Load data
        self.stdout.write('Loading patient data...')
        df = loader.load_from_queryset()
        n_total = len(df)
        self.stdout.write(f'  Loaded {n_total} patient records\n')
        
        if n_total == 0:
            self.stdout.write(self.style.ERROR('No patient data found. Aborting.'))
            return
        
        # Feature engineering
        self.stdout.write('Engineering features...')
        df = engineer.process_dataframe(df)
        self.stdout.write('  ✓ Features engineered\n')
        
        # Apply deterministic rules
        self.stdout.write('Applying deterministic clinical rules...')
        df['deterministic_override'] = False
        df['deterministic_prob'] = np.nan
        df['deterministic_reason'] = None
        
        for idx, row in df.iterrows():
            has_override, prob, reason = rules_engine.evaluate_patient(row.to_dict())
            df.at[idx, 'deterministic_override'] = has_override
            df.at[idx, 'deterministic_prob'] = prob
            df.at[idx, 'deterministic_reason'] = reason
        
        n_overrides = df['deterministic_override'].sum()
        self.stdout.write(f'  {n_overrides} cases matched deterministic rules\n')
        
        # Filter training data
        exclude_threshold = options['exclude_high_confidence']
        train_mask = ~(
            (df['deterministic_override']) &
            (df['deterministic_prob'] >= exclude_threshold)
        )
        train_df = df[train_mask].copy()
        n_trainable = len(train_df)
        
        self.stdout.write(
            f'Training on {n_trainable}/{n_total} records '
            f'(excluded {n_total - n_trainable} high-confidence rules)\n'
        )
        
        if n_trainable < 10:
            self.stdout.write(
                self.style.WARNING('WARNING: Very few training samples!\n')
            )
        
        # Prepare features and target
        X, missing_summary = engineer.prepare_features(train_df)
        y = train_df['is_cs'].astype(int)
        
        self.stdout.write('Feature missing value summary:')
        for feature, count in sorted(missing_summary.items(), key=lambda x: -x[1])[:5]:
            self.stdout.write(f'  {feature}: {count} missing')
        self.stdout.write()
        
        # Train-test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=options['test_size'],
                random_state=options['random_state'],
                stratify=y
            )
        except ValueError:
            # Fall back to non-stratified if class imbalance prevents stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=options['test_size'],
                random_state=options['random_state']
            )
        
        self.stdout.write(f'Training samples: {len(X_train)}, Test samples: {len(X_test)}\n')
        
        # Build and train model
        self.stdout.write('Building and training ML pipeline...')
        ml_pipeline.build()
        ml_pipeline.train(X_train, y_train)
        self.stdout.write('  ✓ Training complete\n')
        
        # Evaluate
        self.stdout.write('Evaluating model...')
        metrics = ml_pipeline.evaluate(X_test, y_test)
        metrics['n_patients_total'] = int(n_total)
        metrics['n_trainable'] = int(n_trainable)
        metrics['n_train'] = int(len(X_train))
        metrics['n_test'] = int(len(X_test))
        
        self.stdout.write(self.style.SUCCESS('\n=== Model Performance ==='))
        self.stdout.write(f"Accuracy:  {metrics['accuracy']:.3f}")
        self.stdout.write(f"Precision: {metrics['precision']:.3f}")
        self.stdout.write(f"Recall:    {metrics['recall']:.3f}")
        if metrics.get('roc_auc'):
            self.stdout.write(f"ROC-AUC:   {metrics['roc_auc']:.3f}")
        
        self.stdout.write('\nConfusion Matrix:')
        cm = np.array(metrics['confusion_matrix'])
        self.stdout.write(f'  TN: {cm[0,0]:<4} FP: {cm[0,1]:<4}')
        self.stdout.write(f'  FN: {cm[1,0]:<4} TP: {cm[1,1]:<4}')
        
        self.stdout.write('\n' + metrics['classification_report'])
        
        # Feature importances
        importances_df = ml_pipeline.get_feature_importances()
        if not importances_df.empty:
            self.stdout
