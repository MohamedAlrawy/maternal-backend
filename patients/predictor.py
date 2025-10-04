# # patients/predictor.py
# import os
# import json
# import re
# import joblib
# import numpy as np
# import pandas as pd

# from django.conf import settings
# from django.core.exceptions import ObjectDoesNotExist

# from .models import Patient

# # ----------------- Helpers -----------------
# def safe_parse_list(cell):
#     """Return list of strings for various input formats (list, JSON string, comma-separated)."""
#     if cell is None:
#         return []
#     if isinstance(cell, (list, tuple)):
#         return [str(x).strip() for x in cell if str(x).strip() != ""]
#     if isinstance(cell, dict):
#         # unexpected structure: convert values
#         return [str(v).strip() for v in cell.values() if str(v).strip() != ""]
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
#     low = [str(s).lower() for s in lst]
#     for k in keys:
#         for s in low:
#             if k.lower() in s:
#                 return True
#     return False

# # ----------------- Deterministic Rules -----------------
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

# # ----------------- Model path -----------------
# MODEL_PATH = os.path.join(settings.BASE_DIR, 'artifacts', 'cs_pipeline.joblib')

# # ----------------- Main predictor: payload -> prediction -----------------
# def predict_patient_payload(payload: dict, threshold=0.5):
#     """
#     payload: dict with patient-like keys (strings, numbers, lists).
#     Returns dict: {prediction, probability, reason, source} or {'error': ...}
#     """
#     try:
#         # Normalize list-type fields
#         list_fields = ['menternal_medical','obstetric_history','current_pregnancy_menternal','current_pregnancy_fetal','social']
#         for lf in list_fields:
#             if lf in payload:
#                 payload[lf] = safe_parse_list(payload.get(lf))
#             else:
#                 payload[lf] = []

#         # Derived flags needed by rules and features
#         payload['has_placenta_previa'] = contains_any(payload.get('current_pregnancy_menternal', []), ["placenta previa","placenta praevia","placenta previa/abruption"])
#         payload['has_placenta_abruption'] = contains_any(payload.get('current_pregnancy_menternal', []), ["placenta abruption","abruption"])
#         payload['non_cephalic'] = contains_any(payload.get('current_pregnancy_fetal', []), ["breech","non-cephalic","transverse","oblique"])
#         payload['multiple_gestation'] = contains_any(payload.get('current_pregnancy_menternal', []), ["multiple gestation","twin","twins","triplet"])
#         payload['chronic_hypertension'] = contains_any(payload.get('menternal_medical', []), ["chronic hypertension","hypertension"])
#         payload['diabetes_any'] = contains_any(payload.get('menternal_medical', []), ["diabetes","gdm","gestational diabetes"])
#         payload['gdm'] = contains_any(payload.get('current_pregnancy_menternal', []), ["gdm","gestational diabetes"])
#         payload['history_preeclampsia'] = contains_any(payload.get('obstetric_history', []), ["preeclampsia","pre-eclampsia","pre eclampsia"])

#         # numeric conversions if present
#         for k in ['age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','cervical_dilatation_at_admission','hb_g_dl']:
#             if k in payload:
#                 try:
#                     payload[k] = float(payload[k]) if payload[k] not in (None, '') else None
#                 except Exception:
#                     payload[k] = None

#         # severe_anemia flag
#         payload['severe_anemia'] = True if (payload.get('hb_g_dl') is not None and payload.get('hb_g_dl') < 7) else False

#         # Deterministic override first
#         override, prob_percent, reason = deterministic_check(payload)
#         if override:
#             return {
#                 'prediction': 'CS',
#                 'probability': float(prob_percent) / 100.0,
#                 'reason': reason,
#                 'source': 'rule'
#             }

#         # If no override, ensure model exists
#         if not os.path.exists(MODEL_PATH):
#             return {'error': f"Model artifact not found at {MODEL_PATH}. Train the model first.", 'source': 'model_missing'}

#         # Load pipeline
#         pipeline = joblib.load(MODEL_PATH)

#         # Define features (must match what was used in training script)
#         numeric_features = ['age','parity','bmi','total_number_of_cs','cervical_dilatation_at_admission','estimated_fetal_weight_by_gm']
#         categorical_features = ['presentation','fetus_number','ctg_category','rupture_duration_hour']
#         binary_features = ['chronic_hypertension','diabetes_any','non_cephalic','multiple_gestation','gdm','history_preeclampsia','severe_anemia']

#         # Build single-row DataFrame for pipeline input
#         rec = {}
#         for f in numeric_features + categorical_features + binary_features:
#             rec[f] = payload.get(f, None)

#         # Normalize binary features to 0/1
#         for b in binary_features:
#             val = rec.get(b, False)
#             rec[b] = 1 if (str(val).lower() in ['true','1','yes'] or val is True or val == 1) else 0

#         X_row = pd.DataFrame([rec], columns=(numeric_features + categorical_features + binary_features))
#         proba = float(pipeline.predict_proba(X_row)[:,1][0])
#         pred = 'CS' if proba >= threshold else 'NVD'

#         return {
#             'prediction': pred,
#             'probability': proba,
#             'reason': 'model',
#             'source': 'model'
#         }

#     except Exception as e:
#         return {'error': f'Prediction failed: {str(e)}', 'source': 'predict'}

# # ----------------- Convert Patient instance -> payload -----------------
# def patient_to_payload(patient: Patient) -> dict:
#     payload = {}

#     # Copy simple attributes (exists guard)
#     simple_fields = [
#         'name','file_number','patient_id','nationality','age','height','weight','bmi',
#         'gravidity','parity','abortion','lmp','edd','gestational_age','booking',
#         'blood_group','pulse','bp','temp','oxygen_sat','case_type','room','stage','duration',
#         'surgeon','surgery_type','scheduled_time','time_of_admission','cervical_dilatation_at_admission',
#         'time_of_cervix_fully_dilated','time_of_delivery','labor_duration_hours','fully_dilated_cervix',
#         'total_number_of_cs','cs_indication','presentation','fetus_number','ga_interval',
#         'type_of_labor','mode_of_delivery','type_of_cs','robson_classification','type_of_anasthesia',
#         'blood_loss','indication_of_induction','induction_method','cervix_favrable_for_induction',
#         'membrane_status','rupture_duration_hour','liquor','liquor_2','ctg_category','doctor_name',
#         'hb_g_dl','platelets_x10e9l','estimated_fetal_weight_by_gm','placenta_location'
#     ]
#     for f in simple_fields:
#         try:
#             v = getattr(patient, f)
#             if v is None:
#                 payload[f] = None
#             elif hasattr(v, 'isoformat'):
#                 payload[f] = v.isoformat()
#             else:
#                 payload[f] = v
#         except Exception:
#             payload[f] = None

#     # JSON/list fields
#     json_list_cols = ['menternal_medical','obstetric_history','current_pregnancy_menternal','current_pregnancy_fetal','social']
#     for col in json_list_cols:
#         raw = getattr(patient, col, None)
#         payload[col] = safe_parse_list(raw)

#     # normalize numeric fields
#     for k in ['age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','cervical_dilatation_at_admission','hb_g_dl']:
#         try:
#             if payload.get(k) is not None:
#                 payload[k] = float(payload[k])
#         except Exception:
#             payload[k] = None

#     return payload

# # ----------------- Predict by DB identifier -----------------
# def predict_patient_by_identifier(identifier, threshold=0.5):
#     """
#     Identifier can be PK (int or numeric string), file_number, or patient_id.
#     Returns same dict as predict_patient_payload or error dict.
#     """
#     patient = None
#     try:
#         # Try PK if numeric
#         if isinstance(identifier, int) or (isinstance(identifier, str) and str(identifier).isdigit()):
#             try:
#                 patient = Patient.objects.get(pk=int(identifier))
#             except ObjectDoesNotExist:
#                 patient = None

#         # Try file_number
#         if patient is None:
#             try:
#                 patient = Patient.objects.get(file_number=identifier)
#             except Exception:
#                 patient = None

#         # Try patient_id
#         if patient is None:
#             try:
#                 patient = Patient.objects.get(patient_id=identifier)
#             except Exception:
#                 patient = None

#     except Exception as e:
#         return {'error': f'Error looking up Patient: {str(e)}', 'source': 'db'}

#     if patient is None:
#         return {'error': f'Patient not found for identifier: {identifier}', 'status': 404, 'source': 'not_found'}

#     payload = patient_to_payload(patient)
#     result = predict_patient_payload(payload, threshold=threshold)

#     # attach reference info where available
#     if isinstance(result, dict) and 'error' not in result:
#         result['patient_pk'] = patient.pk
#         result['file_number'] = getattr(patient, 'file_number', None)

#     return result


"""
Cesarean Section (CS) Prediction Module

This module provides prediction capabilities for CS delivery outcomes using:
1. Deterministic clinical rules (evidence-based thresholds)
2. Machine learning model (Random Forest pipeline)

The system prioritizes deterministic rules for high-confidence scenarios,
falling back to ML predictions for ambiguous cases.
"""

import os
import json
import re
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from .models import Patient


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = os.path.join(settings.BASE_DIR, 'artifacts', 'cs_pipeline.joblib')

# Features used by the ML model (must match training configuration)
NUMERIC_FEATURES = [
    'age', 'parity', 'bmi', 'total_number_of_cs',
    'cervical_dilatation_at_admission', 'estimated_fetal_weight_by_gm'
]

CATEGORICAL_FEATURES = [
    'presentation', 'fetus_number', 'ctg_category', 'rupture_duration_hour'
]

BINARY_FEATURES = [
    'chronic_hypertension', 'diabetes_any', 'non_cephalic',
    'multiple_gestation', 'gdm', 'history_preeclampsia', 'severe_anemia'
]

# All features combined
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


# =============================================================================
# DATA PARSING UTILITIES
# =============================================================================

def safe_parse_list(cell: Any) -> List[str]:
    """
    Parse various input formats into a standardized list of strings.
    
    Handles:
    - JSON arrays: '["item1", "item2"]'
    - Comma/semicolon separated: "item1, item2; item3"
    - Python lists/tuples: ['item1', 'item2']
    - Dictionaries: extracts values
    - None/empty values
    
    Args:
        cell: Input value of any type
        
    Returns:
        List of cleaned, non-empty strings
        
    Example:
        >>> safe_parse_list('["diabetes", "hypertension"]')
        ['diabetes', 'hypertension']
        >>> safe_parse_list('diabetes, hypertension; gdm')
        ['diabetes', 'hypertension', 'gdm']
    """
    if cell is None:
        return []
    
    # Already a list or tuple
    if isinstance(cell, (list, tuple)):
        return [str(x).strip() for x in cell if str(x).strip()]
    
    # Dictionary: extract values
    if isinstance(cell, dict):
        return [str(v).strip() for v in cell.values() if str(v).strip()]
    
    # Try parsing as JSON
    if isinstance(cell, str):
        try:
            parsed = json.loads(cell)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Parse as delimited string
        parts = re.split(r'[;,|]\s*', cell)
        return [p.strip() for p in parts if p.strip()]
    
    return []


def contains_any(lst: List[str], keywords: List[str]) -> bool:
    """
    Check if any keyword exists in any list item (case-insensitive substring match).
    
    Args:
        lst: List of strings to search within
        keywords: List of keywords to search for
        
    Returns:
        True if any keyword found in any list item
        
    Example:
        >>> contains_any(['Chronic Hypertension', 'Diabetes'], ['hypertension'])
        True
        >>> contains_any(['Normal pregnancy'], ['diabetes', 'hypertension'])
        False
    """
    if not lst:
        return False
    
    lowercase_items = [str(s).lower() for s in lst]
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        for item in lowercase_items:
            if keyword_lower in item:
                return True
    
    return False


def safe_numeric_conversion(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Safely convert a value to float, returning default if conversion fails.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails (None by default)
        
    Returns:
        Float value or default
    """
    if value is None or value == '':
        return default
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def compute_derived_flags(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute clinical flags from raw patient data.
    
    These flags are used by both deterministic rules and ML features.
    They represent clinically significant conditions extracted from
    list-based fields (medical history, pregnancy complications, etc.).
    
    Args:
        payload: Patient data dictionary
        
    Returns:
        Updated payload with derived flags
        
    Derived Flags:
        - has_placenta_previa: Placenta covering cervix (near 100% CS)
        - has_placenta_abruption: Placental separation (80-90% CS)
        - non_cephalic: Non-head-down presentation (85-95% CS)
        - multiple_gestation: Twins/triplets (50-70% CS)
        - chronic_hypertension: Pre-existing hypertension (40-60% CS)
        - diabetes_any: Any diabetes type (35-55% CS)
        - gdm: Gestational diabetes specifically
        - history_preeclampsia: Previous preeclampsia
        - severe_anemia: Hemoglobin < 7 g/dL (increases CS risk)
    """
    # Extract list fields with safe parsing
    maternal_medical = safe_parse_list(payload.get('menternal_medical', []))
    obstetric_history = safe_parse_list(payload.get('obstetric_history', []))
    current_pregnancy_maternal = safe_parse_list(payload.get('current_pregnancy_menternal', []))
    current_pregnancy_fetal = safe_parse_list(payload.get('current_pregnancy_fetal', []))
    
    # Placental complications (absolute/near-absolute CS indications)
    payload['has_placenta_previa'] = contains_any(
        current_pregnancy_maternal,
        ["placenta previa", "placenta praevia", "placenta previa/abruption"]
    )
    
    payload['has_placenta_abruption'] = contains_any(
        current_pregnancy_maternal,
        ["placenta abruption", "abruption"]
    )
    
    # Fetal presentation
    payload['non_cephalic'] = contains_any(
        current_pregnancy_fetal,
        ["breech", "non-cephalic", "transverse", "oblique"]
    )
    
    # Multiple gestation
    payload['multiple_gestation'] = contains_any(
        current_pregnancy_maternal,
        ["multiple gestation", "twin", "twins", "triplet"]
    )
    
    # Maternal medical conditions
    payload['chronic_hypertension'] = contains_any(
        maternal_medical,
        ["chronic hypertension", "hypertension"]
    )
    
    payload['diabetes_any'] = contains_any(
        maternal_medical,
        ["diabetes", "gdm", "gestational diabetes"]
    )
    
    payload['gdm'] = contains_any(
        current_pregnancy_maternal,
        ["gdm", "gestational diabetes"]
    )
    
    # Obstetric history
    payload['history_preeclampsia'] = contains_any(
        obstetric_history,
        ["preeclampsia", "pre-eclampsia", "pre eclampsia"]
    )
    
    # Severe anemia flag
    hb_value = safe_numeric_conversion(payload.get('hb_g_dl'))
    payload['severe_anemia'] = (hb_value is not None and hb_value < 7)
    
    return payload


def normalize_numeric_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert numeric fields to float, handling missing/invalid values.
    
    Args:
        payload: Patient data dictionary
        
    Returns:
        Updated payload with normalized numeric fields
    """
    numeric_fields = [
        'age', 'parity', 'bmi', 'total_number_of_cs',
        'estimated_fetal_weight_by_gm', 'cervical_dilatation_at_admission',
        'hb_g_dl'
    ]
    
    for field in numeric_fields:
        if field in payload:
            payload[field] = safe_numeric_conversion(payload[field])
    
    return payload


def normalize_binary_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert binary fields to 0/1 integers for model input.
    
    Handles various representations: True/False, 'yes'/'no', 1/0, 'true'/'false'
    
    Args:
        payload: Patient data dictionary
        
    Returns:
        Updated payload with normalized binary fields (0 or 1)
    """
    for field in BINARY_FEATURES:
        value = payload.get(field, False)
        
        # Convert to integer 0 or 1
        if isinstance(value, bool):
            payload[field] = 1 if value else 0
        elif isinstance(value, (int, float)):
            payload[field] = 1 if value == 1 else 0
        elif isinstance(value, str):
            payload[field] = 1 if value.lower() in ['true', '1', 'yes'] else 0
        else:
            payload[field] = 0
    
    return payload


# =============================================================================
# DETERMINISTIC CLINICAL RULES
# =============================================================================

# Evidence-based clinical rules with probability estimates
# Format: (condition_function, probability_percentage, clinical_reason)
CLINICAL_RULES = [
    # Previous cesarean sections (strongest predictors)
    (lambda r: r.get('total_number_of_cs', 0) == 1, 
     35, "Previous 1 CS → ~30-40% repeat CS"),
    
    (lambda r: r.get('total_number_of_cs', 0) == 2, 
     90, "Two prior CS → ≈90% repeat CS"),
    
    (lambda r: r.get('total_number_of_cs', 0) >= 3, 
     99, "Three or more CS → ≈100% CS"),
    
    # Absolute/near-absolute indications
    (lambda r: r.get('has_placenta_previa', False), 
     99, "Placenta previa → ≈100% CS"),
    
    (lambda r: contains_any(r.get('obstetric_history', []), ["uterine rupture"]), 
     99, "Prior uterine rupture → ≈100% CS"),
    
    # High-probability indications
    (lambda r: r.get('has_placenta_abruption', False), 
     85, "Placenta abruption → 80-90% CS"),
    
    (lambda r: r.get('non_cephalic', False), 
     90, "Non-cephalic presentation → 85-95% CS"),
    
    (lambda r: contains_any(r.get('current_pregnancy_menternal', []), 
                            ["ivf", "icsi", "pregnancy post ivf", "pregnancy post icsi"]), 
     85, "Post IVF/ICSI → 80-90% CS"),
    
    # Fetal monitoring concerns
    (lambda r: 'category_iii' in str(r.get('ctg_category', '')).lower() or 
               'category iii' in str(r.get('ctg_category', '')).lower(), 
     95, "CTG Category III → 90-100% CS"),
    
    (lambda r: 'category_ii' in str(r.get('ctg_category', '')).lower() or 
               'category ii' in str(r.get('ctg_category', '')).lower(), 
     70, "CTG Category II → 70% CS"),
    
    # Multiple pregnancy
    (lambda r: r.get('multiple_gestation', False), 
     60, "Multiple gestation → 50-70% CS"),
    
    # Maternal-fetal conditions
    (lambda r: r.get('estimated_fetal_weight_by_gm', 0) >= 4000, 
     50, "Estimated fetal weight ≥4000g → 40-60% CS"),
    
    (lambda r: r.get('chronic_hypertension', False), 
     50, "Chronic hypertension → 40-60% CS"),
    
    (lambda r: r.get('diabetes_any', False), 
     45, "Diabetes → 35-55% CS"),
    
    (lambda r: 'pre_eclampsia_eclampsia_hellp' in str(r.get('cs_indication', '')).lower() or 
               r.get('history_preeclampsia', False), 
     60, "Preeclampsia/HELLP → 50-70% CS"),
    
    (lambda r: contains_any(r.get('menternal_medical', []), ["cardiac disease", "cardiac"]), 
     45, "Cardiac disease → 30-60% CS"),
    
    # Special populations
    (lambda r: contains_any(r.get('menternal_medical', []), ["hiv", "immunocompromised"]), 
     20, "HIV/immunocompromised → 15-25% CS"),
    
    # Severe anemia (lower threshold for intervention)
    (lambda r: not pd.isna(r.get('hb_g_dl')) and float(r.get('hb_g_dl', 999)) < 7, 
     25, "Severe anemia (Hb<7) → 20-30% CS"),
]


def check_deterministic_rules(payload: Dict[str, Any]) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Apply evidence-based clinical rules to determine if CS is highly likely.
    
    Rules are based on obstetric guidelines and literature. The system
    evaluates all applicable rules and returns the highest-probability match.
    
    Args:
        payload: Patient data dictionary with derived flags
        
    Returns:
        Tuple of (rule_triggered, probability, clinical_reason)
        - rule_triggered: True if any rule applies
        - probability: Probability as decimal (0.0-1.0), None if no rule
        - clinical_reason: Human-readable explanation, None if no rule
        
    Example:
        >>> payload = {'total_number_of_cs': 2, ...}
        >>> check_deterministic_rules(payload)
        (True, 0.90, "Two prior CS → ≈90% repeat CS")
    """
    matched_rules = []
    
    for condition_func, prob_percent, reason in CLINICAL_RULES:
        try:
            if condition_func(payload):
                matched_rules.append((prob_percent, reason))
        except Exception:
            # Silently skip rules that fail (e.g., missing data)
            continue
    
    if not matched_rules:
        return False, None, None
    
    # Return the highest-probability rule
    best_match = max(matched_rules, key=lambda x: x[0])
    prob_percent, reason = best_match
    
    return True, prob_percent / 100.0, reason


# =============================================================================
# MACHINE LEARNING PREDICTION
# =============================================================================

def validate_required_features(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Check if payload contains sufficient non-null features for prediction.
    
    A prediction is considered unreliable if too many critical features are missing.
    We require at least 70% of numeric features and 50% of categorical features.
    
    Args:
        payload: Patient data dictionary
        
    Returns:
        Tuple of (is_valid, list_of_missing_critical_features)
    """
    missing_features = []
    
    # Check critical numeric features
    critical_numeric = ['age', 'parity', 'bmi', 'total_number_of_cs']
    numeric_present = sum(1 for f in critical_numeric if payload.get(f) is not None)
    
    if numeric_present < len(critical_numeric) * 0.7:
        missing_features.extend([f for f in critical_numeric if payload.get(f) is None])
    
    # Check critical categorical features
    critical_categorical = ['presentation', 'ctg_category']
    categorical_present = sum(1 for f in critical_categorical if payload.get(f) is not None)
    
    if categorical_present < len(critical_categorical) * 0.5:
        missing_features.extend([f for f in critical_categorical if payload.get(f) is None])
    
    is_valid = len(missing_features) == 0
    return is_valid, missing_features


def load_ml_model() -> Optional[Any]:
    """
    Load the trained ML pipeline from disk.
    
    Returns:
        Loaded sklearn pipeline or None if file doesn't exist
    """
    if not os.path.exists(MODEL_PATH):
        return None
    
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def predict_with_ml_model(payload: Dict[str, Any], pipeline: Any, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Generate prediction using the trained ML model.
    
    Args:
        payload: Patient data with normalized features
        pipeline: Trained sklearn pipeline
        threshold: Probability threshold for CS prediction (default 0.5)
        
    Returns:
        Dictionary with prediction results:
        {
            'prediction': 'CS' or 'NVD',
            'probability': float (0.0-1.0),
            'reason': 'model',
            'source': 'model'
        }
    """
    # Build feature dictionary
    feature_dict = {}
    for feature in ALL_FEATURES:
        feature_dict[feature] = payload.get(feature, None)
    
    # Create single-row DataFrame
    X = pd.DataFrame([feature_dict], columns=ALL_FEATURES)
    
    # Get probability prediction
    proba = float(pipeline.predict_proba(X)[:, 1][0])
    
    # Apply threshold
    prediction = 'CS' if proba >= threshold else 'NVD'
    try:
        clf = pipeline.named_steps['classifier']
        preproc = pipeline.named_steps['preprocessor']
        ohe = preproc.named_transformers_['cat'].named_steps['ohe']
        cat_names = ohe.get_feature_names_out(categorical_features)
        feature_names = numeric_features + list(cat_names) + binary_features

        importances = clf.feature_importances_
        feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

        top_features = [f"{name} ({imp:.2f})" for name, imp in feat_imp[:3]]
        reason = f"Top influencing factors: {', '.join(top_features)}"
    except Exception:
        reason = "Model-based prediction (RandomForest)"
        
    return {
        'prediction': prediction,
        'probability': proba,
        'reason': reason,
        'source': 'model'
    }


# =============================================================================
# MAIN PREDICTION FUNCTION
# =============================================================================

def predict_patient_payload(payload: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
    """
    Main prediction function: takes patient data and returns CS prediction.
    
    Prediction Strategy:
    1. Normalize and parse input data
    2. Compute derived clinical flags
    3. Check deterministic rules (evidence-based guidelines)
    4. If no rule triggers, use ML model
    5. Validate data quality before ML prediction
    
    Args:
        payload: Dictionary with patient data (can be from API, DB, or file)
        threshold: Probability threshold for ML model (default 0.5)
        
    Returns:
        Dictionary with prediction results:
        {
            'prediction': 'CS' or 'NVD',
            'probability': float (0.0-1.0),
            'reason': explanation string or 'model',
            'source': 'rule' or 'model'
        }
        
        OR error dictionary:
        {
            'error': error message,
            'source': error source,
            'missing_features': list (optional)
        }
    """
    try:
        # Step 1: Normalize list fields
        list_fields = [
            'menternal_medical', 'obstetric_history',
            'current_pregnancy_menternal', 'current_pregnancy_fetal', 'social'
        ]
        for field in list_fields:
            if field in payload:
                payload[field] = safe_parse_list(payload[field])
            else:
                payload[field] = []
        
        # Step 2: Compute derived clinical flags
        payload = compute_derived_flags(payload)
        
        # Step 3: Normalize numeric fields
        payload = normalize_numeric_fields(payload)
        
        # Step 4: Check deterministic rules first (highest priority)
        rule_triggered, probability, reason = check_deterministic_rules(payload)
        
        if rule_triggered:
            return {
                'prediction': 'CS',
                'probability': probability,
                'reason': reason,
                'source': 'rule'
            }
        
        # Step 5: Validate data quality for ML prediction
        is_valid, missing_features = validate_required_features(payload)
        
        if not is_valid:
            return {
                'error': 'Insufficient data for prediction. Critical features are missing.',
                'source': 'validation',
                'missing_features': missing_features
            }
        
        # Step 6: Load ML model
        pipeline = load_ml_model()
        
        if pipeline is None:
            return {
                'error': f'ML model not found at {MODEL_PATH}. Please train the model first.',
                'source': 'model_missing'
            }
        
        # Step 7: Normalize binary fields for ML input
        payload = normalize_binary_fields(payload)
        
        # Step 8: Generate ML prediction
        result = predict_with_ml_model(payload, pipeline, threshold)
        
        return result
        
    except Exception as e:
        return {
            'error': f'Prediction failed: {str(e)}',
            'source': 'predict',
            'exception_type': type(e).__name__
        }


# =============================================================================
# DATABASE INTEGRATION
# =============================================================================

def patient_to_payload(patient: Patient) -> Dict[str, Any]:
    """
    Convert Django Patient model instance to prediction payload dictionary.
    
    Args:
        patient: Patient model instance from database
        
    Returns:
        Dictionary with all patient fields formatted for prediction
    """
    payload = {}
    
    # Simple fields to copy directly
    simple_fields = [
        'name', 'file_number', 'patient_id', 'nationality', 'age', 'height', 'weight', 'bmi',
        'gravidity', 'parity', 'abortion', 'lmp', 'edd', 'gestational_age', 'booking',
        'blood_group', 'pulse', 'bp', 'temp', 'oxygen_sat', 'case_type', 'room', 'stage',
        'duration', 'surgeon', 'surgery_type', 'scheduled_time', 'time_of_admission',
        'cervical_dilatation_at_admission', 'time_of_cervix_fully_dilated', 'time_of_delivery',
        'labor_duration_hours', 'fully_dilated_cervix', 'total_number_of_cs', 'cs_indication',
        'presentation', 'fetus_number', 'ga_interval', 'type_of_labor', 'mode_of_delivery',
        'type_of_cs', 'robson_classification', 'type_of_anasthesia', 'blood_loss',
        'indication_of_induction', 'induction_method', 'cervix_favrable_for_induction',
        'membrane_status', 'rupture_duration_hour', 'liquor', 'liquor_2', 'ctg_category',
        'doctor_name', 'hb_g_dl', 'platelets_x10e9l', 'estimated_fetal_weight_by_gm',
        'placenta_location'
    ]
    
    for field in simple_fields:
        try:
            value = getattr(patient, field, None)
            
            if value is None:
                payload[field] = None
            elif hasattr(value, 'isoformat'):  # DateTime objects
                payload[field] = value.isoformat()
            else:
                payload[field] = value
        except AttributeError:
            payload[field] = None
    
    # JSON/list fields
    list_fields = [
        'menternal_medical', 'obstetric_history',
        'current_pregnancy_menternal', 'current_pregnancy_fetal', 'social'
    ]
    
    for field in list_fields:
        raw_value = getattr(patient, field, None)
        payload[field] = safe_parse_list(raw_value)
    
    # Ensure numeric fields are properly typed
    payload = normalize_numeric_fields(payload)
    
    return payload


def predict_patient_by_identifier(
    identifier: Union[int, str],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Predict CS outcome for a patient using database identifier.
    
    Supports multiple identifier types:
    - Primary key (integer ID)
    - File number (string)
    - Patient ID (string)
    
    Args:
        identifier: Patient PK, file_number, or patient_id
        threshold: Probability threshold for prediction
        
    Returns:
        Prediction result dictionary with added fields:
        - patient_pk: Database primary key
        - file_number: Patient file number
        
        OR error dictionary if patient not found
    """
    patient = None
    
    try:
        # Try primary key lookup
        if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
            try:
                patient = Patient.objects.get(pk=int(identifier))
            except ObjectDoesNotExist:
                pass
        
        # Try file_number lookup
        if patient is None:
            try:
                patient = Patient.objects.get(file_number=identifier)
            except (ObjectDoesNotExist, Exception):
                pass
        
        # Try patient_id lookup
        if patient is None:
            try:
                patient = Patient.objects.get(patient_id=identifier)
            except (ObjectDoesNotExist, Exception):
                pass
        
    except Exception as e:
        return {
            'error': f'Database error while looking up patient: {str(e)}',
            'source': 'db'
        }
    
    # Patient not found
    if patient is None:
        return {
            'error': f'Patient not found for identifier: {identifier}',
            'status': 404,
            'source': 'not_found'
        }
    
    # Convert to payload and predict
    payload = patient_to_payload(patient)
    result = predict_patient_payload(payload, threshold=threshold)
    
    # Attach reference information
    if isinstance(result, dict) and 'error' not in result:
        result['patient_pk'] = patient.pk
        result['file_number'] = getattr(patient, 'file_number', None)
        result['patient_id'] = getattr(patient, 'patient_id', None)
    
    return result
