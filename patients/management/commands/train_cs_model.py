"""
Updated CS training command with obstetric_history -> total_number_of_cs mapping.

This file is intended to replace the existing training command logic.
It adds explicit handling of obstetric_history tokens:
 - "Previous c-section" or variations -> total_number_of_cs >= 1
 - "Multiple c-sections (2)" or variations -> total_number_of_cs >= 2
 - "Multiple c-sections (>3)" or variations -> total_number_of_cs >= 3

Placement (recommended):
    patients/management/commands/train_cs.py
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
except Exception:
    Patient = None

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    numeric: List[str]
    categorical: List[str]
    binary: List[str]
    list_columns: List[str]
    @property
    def all_features(self) -> List[str]:
        return self.numeric + self.categorical + self.binary


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
        'current_pregnancy_menternal', 'current_pregnancy_fetal', 'social', 'liquor'
    ]
)


class DataParser:
    @staticmethod
    def parse_list(cell: Any) -> List[str]:
        if cell is None:
            return []
        if isinstance(cell, (list, tuple)):
            return [str(x).strip() for x in cell if str(x).strip()]
        if isinstance(cell, dict):
            return [str(v).strip() for v in cell.values() if str(v).strip()]
        if isinstance(cell, str):
            try:
                parsed = json.loads(cell)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
            parts = re.split(r'[;,|]\s*', cell)
            return [p.strip() for p in parts if p.strip()]
        return []

    @staticmethod
    def contains_any(items: List[str], keywords: List[str]) -> bool:
        if not items:
            return False
        items_lower = [str(s).lower() for s in items]
        return any(keyword.lower() in item for keyword in keywords for item in items_lower)


class ClinicalRule:
    def __init__(self, condition: Callable[[Dict], bool], probability: float, reason: str):
        self.condition = condition
        self.probability = probability
        self.reason = reason

    def evaluate(self, patient_data: Dict) -> Tuple[bool, Optional[float], Optional[str]]:
        try:
            if self.condition(patient_data):
                return True, self.probability, self.reason
        except Exception as e:
            logger.debug(f"Rule error: {e}")
        return False, None, None


class ClinicalRulesEngine:
    def __init__(self):
        self.rules = self._define_rules()

    def _define_rules(self) -> List[ClinicalRule]:
        return [
            ClinicalRule(lambda r: r.get('total_number_of_cs', 0) == 1, 35, "Previous 1 CS → 30-40% repeat CS"),
            ClinicalRule(lambda r: r.get('total_number_of_cs', 0) == 2, 90, "Two prior CS → ≈90% repeat CS"),
            ClinicalRule(lambda r: r.get('total_number_of_cs', 0) >= 3, 99, "Three or more CS → ≈100% CS"),
            ClinicalRule(lambda r: r.get('has_placenta_previa', False), 99, "Placenta previa → ≈100% CS"),
            ClinicalRule(lambda r: r.get('has_placenta_abruption', False), 85, "Placenta abruption → 80-90% CS"),
            ClinicalRule(lambda r: r.get('non_cephalic', False), 90, "Non-cephalic presentation → 85-95% CS"),
            ClinicalRule(lambda r: r.get('multiple_gestation', False), 60, "Multiple gestation → 50-70% CS"),
            ClinicalRule(lambda r: r.get('estimated_fetal_weight_by_gm', 0) >= 4000, 50, "Macrosomia (≥4000g) → 40-60% CS"),
            ClinicalRule(lambda r: r.get('chronic_hypertension', False), 50, "Chronic hypertension → 40-60% CS"),
            ClinicalRule(lambda r: r.get('diabetes_any', False), 45, "Diabetes → 35-55% CS"),
            ClinicalRule(lambda r: ('pre_eclampsia_eclampsia_hellp' in str(r.get('cs_indication', '')).lower()) or r.get('history_preeclampsia', False), 60, "Preeclampsia/HELLP → 50-70% CS"),
            ClinicalRule(lambda r: not pd.isna(r.get('hb_g_dl')) and float(r.get('hb_g_dl', 999)) < 7, 25, "Severe anemia (Hb<7) → 20-30% CS"),
            ClinicalRule(lambda r: DataParser.contains_any(r.get('menternal_medical', []), ["cardiac disease", "cardiac"]), 45, "Cardiac disease → 30-60% CS"),
            ClinicalRule(lambda r: DataParser.contains_any(r.get('menternal_medical', []), ["hiv", "immunocompromised"]), 20, "HIV/immunocompromised → 15-25% CS"),
            ClinicalRule(lambda r: DataParser.contains_any(r.get('obstetric_history', []), ["uterine rupture"]), 99, "Prior uterine rupture → ≈100% CS"),
            ClinicalRule(lambda r: ('category_ii' in str(r.get('ctg_category', '')).lower().replace(' ', '_')) or ('category ii' in str(r.get('ctg_category', '')).lower()), 70, "CTG Category II → 70% CS"),
            ClinicalRule(lambda r: ('category_iii' in str(r.get('ctg_category', '')).lower().replace(' ', '_')) or ('category iii' in str(r.get('ctg_category', '')).lower()), 95, "CTG Category III → 90-100% CS"),
            ClinicalRule(lambda r: DataParser.contains_any(r.get('current_pregnancy_menternal', []), ["ivf", "icsi", "pregnancy post ivf", "pregnancy post icsi"]), 85, "Post IVF/ICSI → 80-90% CS"),
        ]

    def evaluate_patient(self, patient_data: Dict) -> Tuple[bool, Optional[float], Optional[str]]:
        matches = []
        for rule in self.rules:
            matched, prob, reason = rule.evaluate(patient_data)
            if matched:
                matches.append((prob, reason))
        if not matches:
            return False, None, None
        best_match = max(matches, key=lambda x: x[0])
        return True, best_match[0], best_match[1]


class PatientDataLoader:
    REQUIRED_FIELDS = [
        'id', 'name', 'file_number', 'patient_id', 'age', 'parity', 'bmi',
        'bp', 'Hb_g_dL', 'hb_g_dl', 'total_number_of_cs', 'presentation',
        'fetus_number', 'cervical_dilatation_at_admission', 'ctg_category',
        'estimated_fetal_weight_by_gm', 'rupture_duration_hour',
        'mode_of_delivery', 'cs_indication', 'menternal_medical',
        'obstetric_history', 'current_pregnancy_menternal',
        'current_pregnancy_fetal', 'social', 'liquor'
    ]

    @staticmethod
    def load_from_queryset(queryset=None) -> pd.DataFrame:
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
    def __init__(self, feature_config: FeatureConfig):
        self.config = feature_config
        self.parser = DataParser()

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._normalize_list_columns(df)
        df = self._derive_clinical_flags(df)
        # apply obstetric_history mapping to total_number_of_cs
        df = self._apply_obstetric_history_cs_mapping(df)
        df = self._normalize_hemoglobin(df)
        df = self._normalize_numeric_columns(df)
        df = self._compute_target(df)
        return df

    def _normalize_list_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.config.list_columns:
            if col not in df.columns:
                df[col] = [[] for _ in range(len(df))]
            else:
                df[col] = df[col].apply(self.parser.parse_list)
        return df

    def _derive_clinical_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        flag_definitions = {
            'has_placenta_previa': ('current_pregnancy_menternal', ["placenta previa", "placenta praevia", "placenta previa/abruption"]),
            'has_placenta_abruption': ('current_pregnancy_menternal', ["placenta abruption", "abruption"]),
            'non_cephalic': ('current_pregnancy_fetal', ["breech", "non-cephalic", "transverse", "oblique"]),
            'multiple_gestation': ('current_pregnancy_menternal', ["multiple gestation", "twin", "twins", "triplet"]),
            'chronic_hypertension': ('menternal_medical', ["chronic hypertension", "hypertension"]),
            'diabetes_any': ('menternal_medical', ["diabetes", "gdm", "gestational diabetes"]),
            'gdm': ('current_pregnancy_menternal', ["gdm", "gestational diabetes"]),
            'history_preeclampsia': ('obstetric_history', ["preeclampsia", "pre-eclampsia", "pre eclampsia"]),
        }
        for flag_name, (source_col, keywords) in flag_definitions.items():
            if source_col not in df.columns:
                df[flag_name] = False
            else:
                df[flag_name] = df[source_col].apply(lambda items: self.parser.contains_any(items, keywords))
        return df

    def _apply_obstetric_history_cs_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map tokens in obstetric_history -> total_number_of_cs
        Recognised tokens:
          - 'previous c-section' / 'previous cs' -> 1
          - 'multiple c-sections (2)' or variants -> 2
          - 'multiple c-sections (>3)' or 'more than 3' -> 3
          - 'multiple c-sections' (no number) -> treat as 2
        This will increase current total_number_of_cs only if mapped value > existing.
        """
        def map_row(o_list, current_cs):
            try:
                mapped = int(current_cs) if (current_cs is not None and not pd.isna(current_cs)) else 0
            except Exception:
                mapped = 0
            low = [str(x).lower() for x in (o_list or [])]
            for item in low:
                if 'previous c-section' in item or 'previous cs' in item or 'previous c section' in item:
                    mapped = max(mapped, 1)
                # explicit "Multiple c-sections (2)" or "multiple c-sections 2"
                if re.search(r'multiple c-?sections\\s*\\(?\\s*2\\s*\\)?', item) or 'multiple c-sections (2)' in item or 'multiple cs (2)' in item:
                    mapped = max(mapped, 2)
                # explicit >3
                if 'multiple c-sections' in item and ('>3' in item or 'more than 3' in item or '(>3)' in item or 'greater than 3' in item):
                    mapped = max(mapped, 3)
                # ambiguous "multiple c-sections" without number -> assume 2
                if 'multiple c-sections' in item and not re.search(r'\\d', item):
                    mapped = max(mapped, 2)
                # some datasets may use "previous cesarean" wording
                if 'previous cesarean' in item:
                    mapped = max(mapped, 1)
            return mapped

        if 'obstetric_history' not in df.columns:
            return df
        # ensure numeric column exists
        if 'total_number_of_cs' not in df.columns:
            df['total_number_of_cs'] = 0
        else:
            df['total_number_of_cs'] = pd.to_numeric(df['total_number_of_cs'], errors='coerce').fillna(0)
        mapped_vals = []
        for idx, row in df.iterrows():
            try:
                o_list = row.get('obstetric_history') or []
                mapped_vals.append(map_row(o_list, row.get('total_number_of_cs', 0)))
            except Exception:
                mapped_vals.append(row.get('total_number_of_cs', 0))
        df['total_number_of_cs'] = mapped_vals
        return df

    def _normalize_hemoglobin(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Hb_g_dL' in df.columns and df['Hb_g_dL'].notna().any():
            df['hb_g_dl'] = pd.to_numeric(df['Hb_g_dL'], errors='coerce')
        else:
            if 'hb_g_dl' not in df.columns:
                df['hb_g_dl'] = np.nan
            else:
                df['hb_g_dl'] = pd.to_numeric(df['hb_g_dl'], errors='coerce')
        df['severe_anemia'] = df['hb_g_dl'].apply(lambda x: not pd.isna(x) and float(x) < 7)
        return df

    def _normalize_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'total_number_of_cs' in df.columns:
            df['total_number_of_cs'] = pd.to_numeric(df['total_number_of_cs'], errors='coerce').fillna(0)
        else:
            df['total_number_of_cs'] = 0
        for col in self.config.numeric:
            if col == 'total_number_of_cs':
                continue
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = np.nan
        return df

    def _compute_target(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'mode_of_delivery' not in df.columns:
            df['mode_of_delivery'] = None
        df['mode_of_delivery_norm'] = df['mode_of_delivery'].fillna('').astype(str).str.lower()
        cs_keywords = ['cesarean', 'cs', 'c/s', 'c.s']
        df['is_cs'] = df['mode_of_delivery_norm'].apply(lambda s: int(any(kw in s for kw in cs_keywords)))
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        X = df.copy()
        missing_summary = {}
        for feature in self.config.all_features:
            if feature not in X.columns:
                X[feature] = np.nan
            missing_summary[feature] = int(X[feature].isna().sum())
        for binary_feature in self.config.binary:
            X[binary_feature] = X[binary_feature].apply(lambda v: int(str(v).lower() in ['true', '1', 'yes'] or v is True or v == 1))
        return X[self.config.all_features], missing_summary


class CSPredictorPipeline:
    def __init__(self, feature_config: FeatureConfig):
        self.config = feature_config
        self.pipeline = None

    def build(self) -> Pipeline:
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, self.config.numeric), ('cat', categorical_transformer, self.config.categorical)], remainder='passthrough')
        classifier = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
        return self.pipeline

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        if self.pipeline is None:
            raise RuntimeError("Pipeline not built. Call build() first.")
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        if self.pipeline is None:
            raise RuntimeError("Pipeline not trained.")
        y_pred = self.pipeline.predict(X_test)
        probs = None
        if hasattr(self.pipeline, 'predict_proba'):
            try:
                probs = self.pipeline.predict_proba(X_test)[:, 1]
            except Exception:
                probs = None
        metrics = {'accuracy': float(accuracy_score(y_test, y_pred)), 'precision': float(precision_score(y_test, y_pred, zero_division=0)), 'recall': float(recall_score(y_test, y_pred, zero_division=0)), 'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(), 'classification_report': classification_report(y_test, y_pred, zero_division=0)}
        if probs is not None and len(np.unique(y_test)) > 1:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_test, probs))
            except Exception:
                metrics['roc_auc'] = None
        else:
            metrics['roc_auc'] = None
        return metrics

    def get_feature_importances(self) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("Pipeline not trained.")
        try:
            preprocessor = self.pipeline.named_steps['preprocessor']
            ohe = preprocessor.named_transformers_['cat'].named_steps['ohe']
            cat_names = ohe.get_feature_names_out(self.config.categorical)
            all_names = list(self.config.numeric) + list(cat_names) + list(self.config.binary)
            importances = self.pipeline.named_steps['classifier'].feature_importances_
            return pd.DataFrame({'feature': all_names, 'importance': importances}).sort_values('importance', ascending=False)
        except Exception as e:
            logger.error(f"Feature importance error: {e}")
            return pd.DataFrame()


class ArtifactsManager:
    def __init__(self, artifacts_dir: Optional[Path] = None):
        if artifacts_dir is None:
            artifacts_dir = Path(settings.BASE_DIR) / 'artifacts'
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def save_pipeline(self, pipeline: Pipeline, filename: str = 'cs_pipeline.joblib') -> Path:
        path = self.artifacts_dir / filename
        joblib.dump(pipeline, path)
        return path

    def save_metrics(self, metrics: Dict, filename: str = 'cs_metrics.json') -> Path:
        path = self.artifacts_dir / filename
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            return obj
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2, default=convert)
        return path

    def save_feature_importances(self, importances_df: pd.DataFrame, filename: str = 'cs_feature_importances.csv') -> Path:
        path = self.artifacts_dir / filename
        importances_df.to_csv(path, index=False)
        return path

    def load_pipeline(self, filename: str = 'cs_pipeline.joblib') -> Pipeline:
        path = self.artifacts_dir / filename
        return joblib.load(path)


class Command(BaseCommand):
    help = 'Train CS predictor with deterministic rules and ML pipeline'

    def add_arguments(self, parser):
        parser.add_argument('--exclude-high-confidence', type=int, default=80, help='Exclude deterministic rule matches >= this probability from training')
        parser.add_argument('--test-size', type=float, default=0.2, help='Fraction of data to use for testing')
        parser.add_argument('--random-state', type=int, default=42, help='Random seed for reproducibility')

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('=== CS Predictor Training ===\n'))
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
        n_overrides = int(df['deterministic_override'].sum())
        self.stdout.write(f'  {n_overrides} cases matched deterministic rules\n')

        exclude_threshold = options['exclude_high_confidence']
        train_mask = ~((df['deterministic_override']) & (df['deterministic_prob'] >= exclude_threshold))
        train_df = df[train_mask].copy()
        n_trainable = len(train_df)
        self.stdout.write(f'Training on {n_trainable}/{n_total} records (excluded {n_total - n_trainable} high-confidence rules)\n')
        if n_trainable < 10:
            self.stdout.write(self.style.WARNING('WARNING: Very few training samples!\n'))

        X, missing_summary = engineer.prepare_features(train_df)
        y = train_df['is_cs'].astype(int)
        self.stdout.write('Feature missing value summary:')
        for feature, count in sorted(missing_summary.items(), key=lambda x: -x[1])[:10]:
            self.stdout.write(f'  {feature}: {count} missing')
        self.stdout.write('')

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=options['test_size'], random_state=options['random_state'], stratify=y)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=options['test_size'], random_state=options['random_state'])

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

        importances_df = ml_pipeline.get_feature_importances()
        if not importances_df.empty:
            artifacts_mgr.save_feature_importances(importances_df)
        artifacts_mgr.save_pipeline(ml_pipeline.pipeline)
        artifacts_mgr.save_metrics(metrics)

        self.stdout.write(self.style.SUCCESS('Artifacts saved. Training complete.'))
