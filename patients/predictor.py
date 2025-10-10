"""
CS predictor: patient -> payload -> prediction
Path: patients/cs_predictor_refactored.py
- Maps obstetric_history tokens to total_number_of_cs (same rules as training)
- Runs deterministic rules first; if matched, returns rule-based prediction with reason
- Otherwise loads pipeline artifact and returns model prediction with probability and reason='model'
- If required numeric inputs are missing for model, returns an error dict
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
    Patient = None

MODEL_FILENAME = 'cs_pipeline.joblib'
MODEL_PATH = os.path.join(getattr(settings, 'BASE_DIR', '.'), 'artifacts', MODEL_FILENAME)

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

def map_cs_from_obstetric_history(obst_list, current_cs):
    mapped = int(current_cs) if (current_cs is not None and not pd.isna(current_cs)) else 0
    low = [str(x).lower() for x in (obst_list or [])]
    for item in low:
        if 'previous c-section' in item or 'previous cs' in item or 'previous c section' in item:
            mapped = max(mapped, 1)
        if re.search(r'multiple c-?sections\\s*\\(?\\s*2\\s*\\)?', item) or 'multiple c-sections (2)' in item or 'multiple cs (2)' in item:
            mapped = max(mapped, 2)
        if 'multiple c-sections' in item and ('>3' in item or 'more than 3' in item or '(>3)' in item):
            mapped = max(mapped, 3)
        if 'multiple c-section' in item and not re.search(r'\\d', item):
            mapped = max(mapped, 2)
    return mapped

def patient_to_payload(patient: Patient) -> dict:
    payload = {}
    simple_fields = [
        'id','name','file_number','patient_id','age','height','weight','bmi',
        'gravidity','parity','abortion','lmp','edd','gestational_age','booking',
        'blood_group','pulse','bp','temp','oxygen_sat','case_type','room','stage','duration',
        'surgeon','surgery_type','scheduled_time','time_of_admission','cervical_dilatation_at_admission',
        'time_of_cervix_fully_dilated','time_of_delivery','labor_duration_hours','fully_dilated_cervix',
        'total_number_of_cs','cs_indication','presentation','fetus_number','ga_interval',
        'type_of_labor','mode_of_delivery','type_of_cs','robson_classification','type_of_anasthesia',
        'blood_loss','indication_of_induction','induction_method','cervix_favrable_for_induction',
        'membrane_status','rupture_duration_hour','liquor','liquor_2','ctg_category','doctor_name',
        'hb_g_dl','platelets_x10e9l','estimated_fetal_weight_by_gm','placenta_location'
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
    # JSON/list fields
    json_list_cols = ['menternal_medical','obstetric_history','current_pregnancy_menternal','current_pregnancy_fetal','social']
    for col in json_list_cols:
        raw = getattr(patient, col, None)
        payload[col] = safe_parse_list(raw)
    # Map CS from obstetric_history if needed
    try:
        current_cs = payload.get('total_number_of_cs', None)
        mapped = map_cs_from_obstetric_history(payload.get('obstetric_history', []), current_cs)
        payload['total_number_of_cs'] = float(mapped)
    except Exception:
        pass
    # numeric conversions
    for k in ['age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','cervical_dilatation_at_admission','hb_g_dl']:
        try:
            if payload.get(k) is not None:
                payload[k] = float(payload[k])
        except Exception:
            payload[k] = None
    return payload

def predict_patient_payload(payload: dict, threshold=0.5):
    try:
        # normalize list-type fields
        list_fields = ['menternal_medical','obstetric_history','current_pregnancy_menternal','current_pregnancy_fetal','social']
        for lf in list_fields:
            payload[lf] = safe_parse_list(payload.get(lf))
        # derived flags needed by rules and features
        payload['has_placenta_previa'] = contains_any(payload.get('current_pregnancy_menternal', []), ['placenta previa','placenta praevia','placenta previa/abruption'])
        payload['has_placenta_abruption'] = contains_any(payload.get('current_pregnancy_menternal', []), ['placenta abruption','abruption'])
        payload['non_cephalic'] = contains_any(payload.get('current_pregnancy_fetal', []), ['breech','non-cephalic','transverse','oblique'])
        payload['multiple_gestation'] = contains_any(payload.get('current_pregnancy_menternal', []), ['multiple gestation','twin','twins','triplet'])
        payload['chronic_hypertension'] = contains_any(payload.get('menternal_medical', []), ['chronic hypertension','hypertension'])
        payload['diabetes_any'] = contains_any(payload.get('menternal_medical', []), ['diabetes','gdm','gestational diabetes'])
        payload['gdm'] = contains_any(payload.get('current_pregnancy_menternal', []), ['gdm','gestational diabetes'])
        payload['history_preeclampsia'] = contains_any(payload.get('obstetric_history', []), ['preeclampsia','pre-eclampsia','pre eclampsia'])
        # ensure total_number_of_cs mapping fallback
        try:
            payload['total_number_of_cs'] = float(payload.get('total_number_of_cs')) if payload.get('total_number_of_cs') not in (None,'') else None
        except Exception:
            payload['total_number_of_cs'] = None
        if not payload.get('total_number_of_cs'):
            payload['total_number_of_cs'] = map_cs_from_obstetric_history(payload.get('obstetric_history', []), 0)
        # numeric conversions for model features
        for k in ['age','parity','bmi','total_number_of_cs','estimated_fetal_weight_by_gm','cervical_dilatation_at_admission','hb_g_dl']:
            if k in payload:
                try:
                    payload[k] = float(payload[k]) if payload[k] not in (None,'') else None
                except Exception:
                    payload[k] = None
        # deterministic rules
        RULES = [
            (lambda r: (r.get('total_number_of_cs', 0) == 1), 35, "Previous 1 CS -> ~30-40% repeat CS"),
            (lambda r: (r.get('total_number_of_cs', 0) == 2), 90, "Two prior CS -> ≈90% repeat CS"),
            (lambda r: (r.get('total_number_of_cs', 0) >= 3), 99, "Three or more CS -> ≈100% CS"),
            (lambda r: r.get('has_placenta_previa', False), 99, "Placenta previa -> ≈100% CS"),
            (lambda r: r.get('has_placenta_abruption', False), 85, "Placenta abruption -> 80-90% CS"),
            (lambda r: r.get('non_cephalic', False), 90, "Non-cephalic presentation -> 85-95% CS"),
            (lambda r: r.get('multiple_gestation', False), 60, "Multiple gestation -> 50-70% CS"),
            (lambda r: (r.get('estimated_fetal_weight_by_gm', 0) >= 4000), 50, "Estimated fetal weight ≥4000g -> 40-60% CS"),
            (lambda r: r.get('chronic_hypertension', False), 50, "Chronic hypertension -> 40-60% CS"),
            (lambda r: r.get('diabetes_any', False), 45, "Diabetes -> 35-55% CS"),
            (lambda r: (('pre_eclampsia_eclampsia_hellp' in str(r.get('cs_indication', '')).lower()) or r.get('history_preeclampsia', False)), 60, "Preeclampsia/HELLP -> 50-70% CS"),
            (lambda r: (not pd.isna(r.get('hb_g_dl')) and float(r.get('hb_g_dl', 999)) < 7), 25, "Severe anemia (Hb<7) -> 20-30% CS"),
            (lambda r: contains_any(r.get('menternal_medical', []), ["cardiac disease", "cardiac"]), 45, "Cardiac disease -> 30-60% CS"),
            (lambda r: contains_any(r.get('menternal_medical', []), ["hiv", "immunocompromised"]), 20, "HIV/immunocompromised -> 15-25%"),
            (lambda r: contains_any(r.get('obstetric_history', []), ["uterine rupture"]), 99, "Prior uterine rupture -> ≈100% CS"),
            (lambda r: (str(r.get('ctg_category','')).lower().find('category_ii')>=0 or str(r.get('ctg_category','')).lower().find('category ii')>=0), 70, "CTG Category II -> 70% CS"),
            (lambda r: (str(r.get('ctg_category','')).lower().find('category_iii')>=0 or str(r.get('ctg_category','')).lower().find('category iii')>=0), 95, "CTG Category III -> 90-100% CS"),
            (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ["ivf", "icsi", "pregnancy post ivf", "pregnancy post icsi"]), 85, "Post IVF/ICSI -> 80-90% CS"),
        ]
        matches = []
        for cond, prob, reason in RULES:
            try:
                if cond(payload):
                    matches.append((prob, reason))
            except Exception:
                continue
        if matches:
            best = max(matches, key=lambda x: x[0])
            return {'prediction': 'CS', 'probability': float(best[0]) / 100.0, 'reason': best[1], 'source': 'rule'}
        # model fallback
        if not os.path.exists(MODEL_PATH):
            return {'error': f"Model artifact not found at {MODEL_PATH}. Train the model first.", 'source': 'model_missing'}
        pipeline = joblib.load(MODEL_PATH)
        numeric_features = ['age','parity','bmi','total_number_of_cs','cervical_dilatation_at_admission','estimated_fetal_weight_by_gm']
        categorical_features = ['presentation','fetus_number','ctg_category','rupture_duration_hour']
        binary_features = ['chronic_hypertension','diabetes_any','non_cephalic','multiple_gestation','gdm','history_preeclampsia','severe_anemia']
        rec = {}
        for f in numeric_features + categorical_features + binary_features:
            rec[f] = payload.get(f, None)
        for b in binary_features:
            val = rec.get(b, False)
            rec[b] = 1 if (str(val).lower() in ['true','1','yes'] or val is True or val == 1) else 0
        X_row = pd.DataFrame([rec], columns=(numeric_features + categorical_features + binary_features))
        try:
            proba = float(pipeline.predict_proba(X_row)[:,1][0])
            pred = 'CS' if proba >= threshold else 'NVD'
            return {'prediction': pred, 'probability': proba, 'reason': 'model', 'source': 'model'}
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}', 'source': 'predict'}

    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}', 'source': 'predict'}

def predict_patient_by_identifier(identifier, threshold=0.5):
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
    result = predict_patient_payload(payload, threshold=threshold)
    if isinstance(result, dict) and 'error' not in result:
        result['patient_pk'] = patient.pk
        result['file_number'] = getattr(patient, 'file_number', None)
    return result
