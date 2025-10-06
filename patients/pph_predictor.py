"""
PPH predictor (direct feature set) - regenerated robust version

Features:
- Accepts identifier (PK/int or numeric string), file_number, patient_id,
  OR a payload dict/list or JSON string representing patient data.
- Applies deterministic clinical rules first (high-confidence short-circuit).
- If rules don't decide, builds feature row from available direct fields and relies on
  model imputers for missing numeric values (tolerant).
- Returns detailed reason: deterministic rule reasons or top feature importances.
- Returns structured errors (status 400) when payload is invalid or insufficient.

Placement: patients/pph_predictor_direct.py
"""

import os
import json
import re
import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist

# Try import Patient model; allow editing/testing outside Django
try:
    from .models import Patient
except Exception:
    Patient = None

MODEL_FILENAME = 'pph_direct_pipeline.joblib'
MODEL_PATH = os.path.join(getattr(settings, 'BASE_DIR', '.'), 'artifacts', MODEL_FILENAME)

# --- Helpers ---
def safe_parse_list(cell):
    """Normalize many encodings into list[str]."""
    if cell is None:
        return []
    if isinstance(cell, (list, tuple)):
        return [str(x).strip() for x in cell if str(x).strip()!='']
    if isinstance(cell, dict):
        return [str(v).strip() for v in cell.values() if str(v).strip()!='']
    if isinstance(cell, str):
        # try JSON
        try:
            parsed = json.loads(cell)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()!='']
            if isinstance(parsed, dict):
                # return dict values as list
                return [str(v).strip() for v in parsed.values() if str(v).strip()!='']
        except Exception:
            pass
        parts = re.split(r'[;,|]\s*', cell)
        return [p.strip() for p in parts if p.strip()!='']
    return []

def contains_any(lst, keys):
    """Case-insensitive substring match between keys and list elements."""
    if not lst:
        return False
    low = [str(s).lower() for s in lst]
    for k in keys:
        kl = k.lower()
        if any(kl in s for s in low):
            return True
    return False

def _ensure_payload_dict(payload):
    """
    Normalize different payload types into a dict.
    Accepts dict, list (wrapped), JSON string, or Django Patient-like object.
    Returns (payload_dict, warnings_list) or raises ValueError for unsupported types.
    """
    warnings = []
    # dict -> ok
    if isinstance(payload, dict):
        return payload, warnings
    # list -> wrap
    if isinstance(payload, list):
        return {'list_payload': payload}, warnings
    # string -> try JSON
    if isinstance(payload, str):
        s = payload.strip()
        if s.startswith('{') or s.startswith('['):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed, warnings
                if isinstance(parsed, list):
                    return {'list_payload': parsed}, warnings
            except Exception:
                raise ValueError("Payload string is not valid JSON")
        raise ValueError("Payload is a plain string (not JSON). Provide dict or JSON string.")
    # Patient-like object -> best-effort attribute extraction
    try:
        if hasattr(payload, '__dict__') or hasattr(payload, 'pk') or hasattr(payload, 'id'):
            keys = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ['menternal_medical','obstetric_history','current_pregnancy_menternal','social','liquor','hb_g_dl','total_number_of_cs','parity']
            d = {}
            for k in keys:
                try:
                    d[k] = getattr(payload, k)
                except Exception:
                    d[k] = None
            return d, warnings
    except Exception:
        pass
    raise ValueError("Unsupported payload type; expected dict, JSON string, list, or Patient instance")

# --- Deterministic rules (weights & reasons) ---
PPH_RULES = [
    (lambda r: contains_any(_safe_get_list(r, 'obstetric_history'), ['postpartum hemorrhage','pph']), 20, "History of PPH (prior)"),
    (lambda r: (r.get('total_number_of_cs', 0) >= 2), 15, "Multiple prior CS"),
    (lambda r: contains_any(_safe_get_list(r, 'current_pregnancy_menternal'), ['placenta previa','placenta praevia']), 20, "Placenta previa"),
    (lambda r: contains_any(_safe_get_list(r, 'current_pregnancy_menternal'), ['placenta abruption','abruption']), 20, "Placental abruption"),
    (lambda r: contains_any(_safe_get_list(r, 'liquor_flags'), ['polyhydramnios','polihydramnios']), 10, "Polyhydramnios"),
    (lambda r: (not pd.isna(r.get('estimated_fetal_weight_by_gm')) and float(r.get('estimated_fetal_weight_by_gm', 0)) > 4000), 10, "Estimated fetal weight >=4000g"),
]

def _safe_get_list(d, key):
    """Return list for key from dict-like payload; tolerate non-dict input."""
    try:
        if not isinstance(d, dict):
            return []
        return safe_parse_list(d.get(key, []))
    except Exception:
        return []

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

# --- Feature lists (must align with training) ---
NUMERIC_FEATURES = ['age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','labor_duration_hours']
CATEGORICAL_FEATURES = ['type_of_labor','mode_of_delivery','placenta_location','rupture_duration_hour','type_of_cs','perineum_integrity','instrumental_delivery']
BINARY_FEATURES = ['history_pph','grand_multipara','multiple_gestation','polihydraminos_flag','placenta_prev_or_abruption','placenta_abruption_flag','severe_anemia','history_transfusion','obstructed_prolonged']

# --- Pipeline loader/builders ---
def load_pipeline(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)

def build_X_row_from_payload(payload):
    """Construct single-row DataFrame from payload; binary features appended as columns."""
    rec = {}
    for k in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        rec[k] = payload.get(k, None)
    X_row = pd.DataFrame([rec], columns=(NUMERIC_FEATURES + CATEGORICAL_FEATURES))
    for b in BINARY_FEATURES:
        val = payload.get(b, False)
        X_row[b] = 1 if (str(val).lower() in ['true','1','yes'] or val is True or val == 1) else 0
    feature_order = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    return X_row[feature_order]

# --- Main prediction functions ---
def predict_pph_direct_payload(raw_payload, threshold=0.5, rule_threshold=80):
    """
    raw_payload: dict, JSON string, list, or Patient-like object.
    Returns dict with keys: prediction, probability, reason, source (or error).
    """
    # Normalize payload
    try:
        payload, warnings = _ensure_payload_dict(raw_payload)
    except ValueError as e:
        return {'error': f'Invalid payload: {str(e)}', 'source': 'payload', 'status': 400}

    # Normalize list fields and derive flags
    list_fields = ['menternal_medical','obstetric_history','current_pregnancy_menternal','social','liquor']
    for lf in list_fields:
        payload[lf] = safe_parse_list(payload.get(lf, []))
    payload['liquor_flags'] = [s.lower() for s in safe_parse_list(payload.get('liquor', [])) if isinstance(s, str)]

    # Derived boolean flags (best-effort)
    payload['history_pph'] = contains_any(payload.get('obstetric_history', []), ['postpartum hemorrhage','pph'])
    payload['grand_multipara'] = contains_any(payload.get('social', []), ['grand multipara']) or (payload.get('parity') is not None and str(payload.get('parity'))!='' and float(payload.get('parity'))>=5)
    payload['multiple_gestation'] = contains_any(payload.get('current_pregnancy_menternal', []), ['multiple gestation','twin','twins','triplet'])
    payload['polihydraminos_flag'] = contains_any(payload.get('liquor_flags', []), ['polyhydramnios','polihydramnios','polihydraminos'])
    payload['placenta_abruption_flag'] = contains_any(payload.get('current_pregnancy_menternal', []), ['placenta abruption','abruption'])
    payload['placenta_prev_or_abruption'] = contains_any(payload.get('current_pregnancy_menternal', []), ['placenta previa','placenta praevia','placenta abruption','abruption'])
    payload['severe_anemia'] = True if (payload.get('hb_g_dl') is not None and str(payload.get('hb_g_dl'))!='' and float(payload.get('hb_g_dl'))<7) else False
    payload['history_transfusion'] = contains_any(payload.get('menternal_medical', []), ['blood transfusion','transfusion'])
    payload['obstructed_prolonged'] = contains_any(payload.get('obstetric_history', []), ['obstructed','prolonged labor','prolonged'])

    # Coerce numeric where possible (tolerant)
    for k in NUMERIC_FEATURES:
        try:
            v = payload.get(k)
            if v in (None, ''):
                payload[k] = None
            else:
                payload[k] = float(v)
        except Exception:
            payload[k] = None

    # Deterministic rules first
    matched, weight, reasons = pph_deterministic_check(payload)
    if matched and weight >= rule_threshold:
        return {'prediction': 'PPH', 'probability': min(0.99, weight/100.0), 'reason': '; '.join(reasons), 'source': 'rule', 'pph': True, 'warnings': warnings if warnings else None}

    # Require at least one numeric and one categorical to proceed (tolerant - but avoid completely empty input)
    has_numeric = any(payload.get(k) not in (None, '') for k in NUMERIC_FEATURES)
    has_categorical = any(payload.get(k) not in (None, '') for k in CATEGORICAL_FEATURES)
    if not has_numeric or not has_categorical:
        missing_fields = [k for k in (NUMERIC_FEATURES + CATEGORICAL_FEATURES) if payload.get(k) in (None, '')]
        return {'error': 'Missing required direct fields for PPH prediction', 'missing_fields': missing_fields, 'status': 400, 'source': 'insufficient_data', 'warnings': warnings if warnings else None}

    # Load pipeline
    try:
        pipeline = load_pipeline()
    except FileNotFoundError as fnf:
        return {'error': str(fnf), 'source': 'model_missing'}

    # Build row and predict
    X_row = build_X_row_from_payload(payload)
    try:
        probs = None
        if hasattr(pipeline, 'predict_proba'):
            probs = pipeline.predict_proba(X_row)
        if probs is not None and probs.ndim == 2 and probs.shape[1] > 1:
            proba = float(probs[:,1][0])
            pred = 'PPH' if proba >= threshold else 'NO_PPH'
            source = 'model'
        else:
            clf = pipeline.named_steps.get('classifier', None)
            if clf is not None and hasattr(clf, 'classes_') and len(clf.classes_) == 1:
                proba = 1.0 if clf.classes_[0] in (1,'1',True) else 0.0
                pred = 'PPH' if proba >= threshold else 'NO_PPH'
                source = 'model'
            else:
                pred_label = pipeline.predict(X_row)[0]
                proba = None
                pred = 'PPH' if int(pred_label) == 1 else 'NO_PPH'
                source = 'model'

        # Reason: deterministic (below threshold) or top feature importances
        if matched and weight > 0 and weight < rule_threshold:
            reason = 'Rule signals: ' + '; '.join(reasons)
        else:
            try:
                clf = pipeline.named_steps['classifier']
                preproc = pipeline.named_steps['preprocessor']
                try:
                    cat_names = list(preproc.get_feature_names_out())
                except Exception:
                    ohe = preproc.named_transformers_['cat'].named_steps['ohe']
                    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
                feature_names = NUMERIC_FEATURES + list(cat_names) + BINARY_FEATURES
                importances = clf.feature_importances_
                feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
                top_features = [f"{name} ({imp:.3f})" for name, imp in feat_imp[:4]]
                reason = f"Top influencing factors: {', '.join(top_features)}"
            except Exception:
                reason = "Model-based prediction"

        return {'prediction': pred, 'probability': proba, 'reason': reason, 'source': source, 'pph': True if pred == 'PPH' else False, 'warnings': warnings if warnings else None}
    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}', 'source': 'predict'}

# --- Identifier helper to accept payloads or lookups ---
def predict_pph_by_identifier(identifier, threshold=0.5, rule_threshold=80):
    """
    Accepts:
      - PK (int) or numeric string
      - file_number (string)
      - patient_id (string)
      - OR a payload dict/list or JSON string (in which case prediction runs on the payload without DB lookup)
    """
    # If identifier looks like payload (dict/list or JSON string starting with '{' or '[') -> treat as payload
    if isinstance(identifier, dict) or isinstance(identifier, list) or (isinstance(identifier, str) and identifier.strip().startswith('{')):
        return predict_pph_direct_payload(identifier, threshold=threshold, rule_threshold=rule_threshold)

    patient = None
    try:
        if isinstance(identifier, int) or (isinstance(identifier, str) and str(identifier).isdigit()):
            try:
                patient = Patient.objects.get(pk=int(identifier))
            except Exception:
                patient = None

        if patient is None:
            try:
                patient = Patient.objects.get(file_number=identifier)
            except Exception:
                patient = None

        if patient is None:
            try:
                patient = Patient.objects.get(patient_id=identifier)
            except Exception:
                patient = None
    except Exception as e:
        return {'error': f'Error looking up Patient: {str(e)}', 'source': 'db'}

    if patient is None:
        return {'error': f'Patient not found for identifier: {identifier}', 'status': 404, 'source': 'not_found'}

    # convert patient object to payload dict and predict
    payload = patient_to_payload(patient)
    return predict_pph_direct_payload(payload, threshold=threshold, rule_threshold=rule_threshold)

# --- Patient -> payload converter ---
def patient_to_payload(patient):
    payload = {}
    simple_fields = ['id','age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','labor_duration_hours','type_of_labor','mode_of_delivery','placenta_location','rupture_duration_hour','type_of_cs','perineum_integrity','instrumental_delivery']
    for f in simple_fields:
        try:
            v = getattr(patient, f)
            payload[f] = None if v is None else (v.isoformat() if hasattr(v, 'isoformat') else v)
        except Exception:
            payload[f] = None
    json_list_cols = ['menternal_medical','obstetric_history','current_pregnancy_menternal','social','liquor']
    for col in json_list_cols:
        raw = getattr(patient, col, None)
        payload[col] = safe_parse_list(raw)
    payload['liquor_flags'] = [s.lower() for s in safe_parse_list(payload.get('liquor', [])) if isinstance(s, str)]
    # derived flags
    payload['history_pph'] = contains_any(payload.get('obstetric_history', []), ['postpartum hemorrhage','pph'])
    payload['grand_multipara'] = contains_any(payload.get('social', []), ['grand multipara']) or (payload.get('parity') is not None and str(payload.get('parity'))!='' and float(payload.get('parity'))>=5)
    payload['multiple_gestation'] = contains_any(payload.get('current_pregnancy_menternal', []), ['multiple gestation','twin','twins','triplet'])
    payload['polihydraminos_flag'] = contains_any(payload.get('liquor_flags', []), ['polyhydramnios','polihydramnios','polihydraminos'])
    payload['placenta_abruption_flag'] = contains_any(payload.get('current_pregnancy_menternal', []), ['placenta abruption','abruption'])
    payload['placenta_prev_or_abruption'] = contains_any(payload.get('current_pregnancy_menternal', []), ['placenta previa','placenta praevia','placenta abruption','abruption'])
    payload['severe_anemia'] = True if (payload.get('hb_g_dl') is not None and str(payload.get('hb_g_dl'))!='' and float(payload.get('hb_g_dl'))<7) else False
    payload['history_transfusion'] = contains_any(payload.get('menternal_medical', []), ['blood transfusion','transfusion'])
    payload['obstructed_prolonged'] = contains_any(payload.get('obstetric_history', []), ['obstructed','prolonged labor','prolonged'])
    # coerce numeric where possible
    for k in NUMERIC_FEATURES:
        try:
            if payload.get(k) is not None and str(payload.get(k))!='':
                payload[k] = float(payload[k])
            else:
                payload[k] = None
        except Exception:
            payload[k] = None
    return payload
