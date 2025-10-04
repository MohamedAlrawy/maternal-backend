"""
PPH Predictor module (refactored)

Place this file in your Django app (patients/) as: patients/pph_predictor_refactored.py

Provides:
 - safe_parse_list, contains_any
 - pph_deterministic_check: deterministic rules with weights and reasons
 - load_pipeline (loads artifacts/pph_pipeline.joblib)
 - build_feature_row: prepares DataFrame for model
 - predict_pph_payload: main entry (returns dict with prediction/probability/reason/source or error)
 - patient_to_payload: convert Patient instance into payload
 - predict_pph_by_identifier: lookup Patient by pk/file_number/patient_id and predict
"""

import os
import json
import re
import joblib
import numpy as np
import pandas as pd

from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist

try:
    from .models import Patient
except Exception:
    # allow static analysis outside Django
    Patient = None

MODEL_FILENAME = 'pph_pipeline.joblib'
MODEL_PATH = os.path.join(getattr(settings, 'BASE_DIR', '.'), 'artifacts', MODEL_FILENAME)

# ---------------- Utilities ----------------
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

# ---------------- Deterministic rules ----------------
PPH_RULES = [
    (lambda r: contains_any(r.get('obstetric_history', []), ['postpartum hemorrhage','pph','post partum hemorrhage']), 20, "History of PPH (prior) -> elevated recurrence risk (15-25%)"),
    (lambda r: ( (isinstance(r.get('social', []), list) and contains_any(r.get('social', []), ['grand multipara'])) or (r.get('parity') is not None and r.get('parity') >= 5) ), 12, "Grand multipara (>=5) -> uterine atony risk (10-15%)"),
    (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['multiple gestation','twin','twins','triplet']), 12, "Multiple gestation -> atony risk (10-15%)"),
    (lambda r: (not pd.isna(r.get('estimated_fetal_weight_by_gm')) and float(r.get('estimated_fetal_weight_by_gm',0)) > 4000), 10, "Estimated fetal weight >=4000g -> overstretching risk (8-12%)"),
    (lambda r: contains_any(r.get('liquor_flags', []), ['polyhydramnios','polihydraminos','polihydramnios']), 9, "Polyhydramnios -> uterine overdistension (8-10%)"),
    (lambda r: r.get('placenta_abruption_flag', False) or contains_any(r.get('current_pregnancy_menternal', []), ['placenta abruption','abruption']), 12, "Placental abruption -> coagulopathy/severe hemorrhage (10-15%)"),
    (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['placenta previa','placenta praevia']), 15, "Placenta previa -> increased bleeding (10-20%)"),
    (lambda r: (not pd.isna(r.get('hb_g_dl')) and float(r.get('hb_g_dl', 999)) < 7), 5, "Severe anemia (Hb<7) - increases morbidity though not bleed amount (flag)"),
    (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['pre-eclampsia','preeclampsia','pre eclampsia']), 7, "Pre-eclampsia -> endothelial dysfunction/coagulopathy (5-10%)"),
    (lambda r: (r.get('total_number_of_cs',0) > 1), 7, "Multiple prior CS (>1) -> adhesions/abnormal placentation (5-10%)"),
    (lambda r: contains_any(r.get('obstetric_history', []), ['obstructed','prolonged labor','prolonged']), 7, "Obstructed/prolonged labor -> uterine exhaustion (5-10%)"),
]

def pph_deterministic_check(rowdict):
    matches = []
    total = 0
    for cond, weight, reason in PPH_RULES:
        try:
            if cond(rowdict):
                matches.append((weight, reason))
                total += weight
        except Exception:
            continue
    if not matches:
        return False, 0, []
    reasons = [r for _, r in matches]
    return True, total, reasons

# ---------------- Model loading ----------------
def load_pipeline(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    return joblib.load(model_path)

# ---------------- Feature builder ----------------
def build_feature_row(payload, required_numeric=None, required_categorical=None):
    if required_numeric is None:
        required_numeric = ['age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','labor_duration_hours']
    if required_categorical is None:
        required_categorical = ['fetus_number','type_of_labor','mode_of_delivery','perineum_integrity','instrumental_delivery','type_of_cs','placenta_location']
    rec = {}
    for k in required_numeric + required_categorical:
        rec[k] = payload.get(k, None)
    df = pd.DataFrame([rec], columns=(required_numeric + required_categorical))
    missing = []
    for n in required_numeric:
        val = df.at[0, n]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            missing.append(n)
    return df, missing

# ---------------- Prediction ----------------
def predict_pph_payload(payload: dict, threshold=0.5):
    try:
        list_fields = ['menternal_medical','obstetric_history','current_pregnancy_menternal','social','liquor']
        for lf in list_fields:
            payload[lf] = safe_parse_list(payload.get(lf, []))
        payload['liquor_flags'] = [s.lower() for s in safe_parse_list(payload.get('liquor', [])) if isinstance(s, str)]
        payload['history_pph'] = contains_any(payload.get('obstetric_history', []), ['postpartum hemorrhage','pph','post partum hemorrhage'])
        payload['grand_multipara'] = contains_any(payload.get('social', []), ['grand multipara']) or (payload.get('parity') is not None and payload.get('parity') >= 5)
        payload['multiple_gestation'] = contains_any(payload.get('current_pregnancy_menternal', []), ['multiple gestation','twin','twins','triplet'])
        payload['placenta_prev_or_abruption'] = contains_any(payload.get('current_pregnancy_menternal', []), ['placenta previa','placenta praevia','placenta abruption','abruption'])
        payload['polihydraminos_flag'] = contains_any(payload.get('liquor_flags', []), ['polihydraminos','polyhydramnios','polihydramnios'])
        payload['placenta_abruption_flag'] = contains_any(payload.get('current_pregnancy_menternal', []), ['placenta abruption','abruption'])
        payload['severe_anemia'] = True if (payload.get('hb_g_dl') is not None and payload.get('hb_g_dl') < 7) else False
        payload['obstructed_prolonged'] = contains_any(payload.get('obstetric_history', []), ['obstructed','prolonged labor','prolonged'])
        for k in ['age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','labor_duration_hours','hb_g_dl','platelets_x10e9l']:
            if k in payload:
                try:
                    payload[k] = float(payload[k]) if payload[k] not in (None, '') else None
                except Exception:
                    payload[k] = None
        matched, weight, reasons = pph_deterministic_check(payload)
        if matched and weight >= 80:
            return {'prediction': 'PPH', 'probability': min(0.99, weight/100.0), 'reason': '; '.join(reasons), 'source': 'rule'}
        try:
            pipeline = load_pipeline()
        except FileNotFoundError as fnf:
            return {'error': str(fnf), 'source': 'model_missing'}
        numeric_features = ['age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','labor_duration_hours']
        categorical_features = ['fetus_number','type_of_labor','mode_of_delivery','perineum_integrity','instrumental_delivery','type_of_cs','placenta_location']
        binary_features = ['chronic_hypertension','history_of_blood_transfusion','history_pph','obstructed_prolonged','multiple_gestation','history_preeclampsia','severe_anemia','grand_multipara','polihydraminos_flag','placenta_prev_or_abruption']
        X_row, missing = build_feature_row(payload, required_numeric=numeric_features, required_categorical=categorical_features)
        if missing:
            return {'error': 'Missing required numeric fields for PPH prediction', 'missing_fields': missing, 'source': 'insufficient_data'}
        for b in binary_features:
            val = payload.get(b, False)
            X_row[b] = 1 if (str(val).lower() in ['true','1','yes'] or val is True or val == 1) else 0
        feature_order = numeric_features + categorical_features + binary_features
        X_row = X_row[feature_order]
        try:
            proba = float(pipeline.predict_proba(X_row)[:,1][0])
            pred = 'PPH' if proba >= threshold else 'NO_PPH'
            reason = None
            if matched and weight > 0:
                reason = '; '.join(reasons)
            else:
                try:
                    clf = pipeline.named_steps['classifier']
                    preproc = pipeline.named_steps['preprocessor']
                    ohe = preproc.named_transformers_['cat'].named_steps['ohe']
                    cat_names = ohe.get_feature_names_out(categorical_features)
                    feature_names = numeric_features + list(cat_names) + binary_features
                    importances = clf.feature_importances_
                    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
                    top_features = [f"{name} ({imp:.3f})" for name, imp in feat_imp[:3]]
                    reason = f"Top influencing factors: {', '.join(top_features)}"
                except Exception:
                    reason = "Model-based prediction"
            return {'prediction': pred, 'probability': proba, 'reason': reason, 'source': 'model'}
        except Exception as e:
            return {'error': f'Prediction failure: {str(e)}', 'source': 'predict'}
    except Exception as e:
        return {'error': f'PPH prediction exception: {str(e)}', 'source': 'predict'}

# ---------------- Patient -> payload ----------------
def patient_to_payload(patient: Patient) -> dict:
    payload = {}
    simple_fields = [
        'name','file_number','patient_id','age','parity','bmi','height','weight',
        'total_number_of_cs','type_of_labor','mode_of_delivery','perineum_integrity','instrumental_delivery','type_of_cs',
        'labor_duration_hours','placenta_location','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','fetus_number'
    ]
    for f in simple_fields:
        try:
            v = getattr(patient, f)
            if v is None:
                payload[f] = None
            elif hasattr(v, 'isoformat'):
                payload[f] = v.isoformat()
            else:
                payload[f] = v
        except Exception:
            payload[f] = None
    json_list_cols = ['menternal_medical','obstetric_history','current_pregnancy_menternal','social','liquor']
    for col in json_list_cols:
        raw = getattr(patient, col, None)
        payload[col] = safe_parse_list(raw)
    for k in ['age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','labor_duration_hours','hb_g_dl','platelets_x10e9l']:
        try:
            if payload.get(k) is not None:
                payload[k] = float(payload[k])
        except Exception:
            payload[k] = None
    return payload

# ---------------- Lookup + predict ----------------
def predict_pph_by_identifier(identifier, threshold=0.5):
    patient = None
    try:
        if isinstance(identifier, int) or (isinstance(identifier, str) and str(identifier).isdigit()):
            try:
                patient = Patient.objects.get(pk=int(identifier))
            except ObjectDoesNotExist:
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
    payload = patient_to_payload(patient)
    result = predict_pph_payload(payload, threshold=threshold)
    if isinstance(result, dict) and 'error' not in result:
        result['patient_pk'] = patient.pk
        result['file_number'] = getattr(patient, 'file_number', None)
    return result
