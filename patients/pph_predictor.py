"""
Updated PPH predictor (direct feature set) with revised deterministic rules.

File: patients/pph_predictor_direct.py
This version updates the deterministic PPH_RULES to match the provided impacts and explanations.
Each rule now returns weight_percent matching the clinical impact ranges. The predictor
continues to short-circuit when high-confidence rules apply, otherwise it uses the saved model.
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
            if isinstance(parsed, dict):
                return [str(v).strip() for v in parsed.values() if str(v).strip()!='']
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
        kl = k.lower()
        if any(kl in s for s in low):
            return True
    return False

def _safe_get_list(d, key):
    try:
        if not isinstance(d, dict):
            return []
        return safe_parse_list(d.get(key, []))
    except Exception:
        return []

def extract_cs_count_from_obstetric(obst_list):
    if not obst_list:
        return 0
    for item in obst_list:
        s=str(item).lower()
        if 'previous c-section' in s or 'previous cs' in s or 'previous c section' in s:
            return 1
        if 'multiple c-sections' in s or 'multiple cs' in s or 'multiple c section' in s:
            # try capture number
            m = re.search(r'(\d+)', s)
            if m:
                try:
                    return int(m.group(1))
                except:
                    pass
            # else return 2 to indicate >1
            return 2
    return 0

# --- Deterministic rules (weights & reasons) updated per user table ---
# Each tuple: (condition_lambda, weight_percent, reason_text)
PPH_RULES = [
    # History of PPH (15-25%)
    (lambda r: contains_any(_safe_get_list(r, 'obstetric_history'), ['postpartum hemorrhage','pph']), 20, "History of postpartum hemorrhage (prior) — recurrence risk 15–25%"),
    # Grand multipara >=5 (10-15%)
    (lambda r: (contains_any(_safe_get_list(r, 'social'), ['grand multipara']) or (r.get('parity') is not None and str(r.get('parity'))!='' and float(r.get('parity'))>=5)), 12, "Grand multipara (>=5) — uterine overdistension, 10–15%"),
    # Multiple gestation in current_pregnancy_menternal (10-15%)
    (lambda r: contains_any(_safe_get_list(r, 'current_pregnancy_menternal'), ['multiple gestation','twin','twins','triplet']), 12, "Multiple gestation (twins/triplets) — 10–15%"),
    # Polyhydramnios in liquor (8-10%)
    (lambda r: contains_any(_safe_get_list(r, 'liquor'), ['polyhydramnios','polihydraminos','polihydramnios']), 9, "Polyhydramnios — uterine overdistension, 8–10%"),
    # Estimated fetal weight > 4000g (8-12%)
    (lambda r: (not pd.isna(r.get('estimated_fetal_weight_by_gm')) and float(r.get('estimated_fetal_weight_by_gm', 0)) > 4000), 10, "Estimated fetal weight >=4000g — macrosomia risk, 8–12%"),
    # History of placenta abruption in obstetric_history (10-15%)
    (lambda r: contains_any(_safe_get_list(r, 'obstetric_history'), ['placenta abruption','abruption']), 12, "History of placenta abruption — coagulopathy/severe hemorrhage, 10–15%"),
    # Placenta previa/abruption in current_pregnancy_menternal (10-20%)
    (lambda r: contains_any(_safe_get_list(r, 'current_pregnancy_menternal'), ['placenta previa','placenta praevia','placenta abruption','abruption']), 15, "Placenta previa/abruption in current pregnancy — 10–20%"),
    # Severe anemia (Hb<7) indirect impact (flag)
    (lambda r: (not pd.isna(r.get('hb_g_dl')) and float(r.get('hb_g_dl', 999)) < 7) or contains_any(_safe_get_list(r, 'current_pregnancy_menternal'), ['severe anemia','hb<7','hb <7']), 3, "Severe anemia (Hb<7) — indirect impact on morbidity (flag)"),
    # Pre-eclampsia (5-10%)
    (lambda r: contains_any(_safe_get_list(r, 'current_pregnancy_menternal'), ['pre-eclampsia','preeclampsia','pre eclampsia']), 7, "Pre-eclampsia — endothelial dysfunction/coagulopathy, 5–10%"),
    # Multiple c-sections (>1) in obstetric_history (5-10%)
    (lambda r: (r.get('total_number_of_cs',0) > 1) or contains_any(_safe_get_list(r, 'obstetric_history'), ['multiple c-sections','multiple cs','two c-sections','2 c-section','>1 c-section']), 7, "Multiple prior C-sections (>1) — adhesions/abnormal placentation, 5–10%"),
    # Obstructed/prolonged labor (5-10%)
    (lambda r: contains_any(_safe_get_list(r, 'obstetric_history'), ['obstructed','prolonged labor','prolonged']), 7, "Obstructed or prolonged labor — uterine exhaustion/atony, 5–10%"),
]

def pph_deterministic_check(rowdict):
    matches = []
    total = 0
    reasons = []
    print(rowdict)
    for cond, weight, reason in PPH_RULES:
        print(cond(rowdict))
        print(cond, weight, reason)
        try:
            if cond(rowdict):
                matches.append({'reason': reason, 'weight_percent': weight})
                total += weight
                reasons.append(reason)
        except Exception:
            continue
    return matches, total, reasons

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
def predict_pph_direct_payload(payload, threshold=0.5, rule_threshold=80):
    # Normalize list fields and derive flags
    list_fields = ['menternal_medical','obstetric_history','current_pregnancy_menternal','social','liquor']
    for lf in list_fields:
        payload[lf] = safe_parse_list(payload.get(lf, []))
    payload['liquor_flags'] = [s.lower() for s in safe_parse_list(payload.get('liquor', [])) if isinstance(s, str)]

    # derived boolean flags
    payload['history_pph'] = contains_any(payload.get('obstetric_history', []), ['postpartum hemorrhage','pph'])
    payload['grand_multipara'] = contains_any(payload.get('social', []), ['grand multipara']) or (payload.get('parity') is not None and str(payload.get('parity'))!='' and float(payload.get('parity'))>=5)
    payload['multiple_gestation'] = contains_any(payload.get('current_pregnancy_menternal', []), ['multiple gestation','twin','twins','triplet'])
    payload['polihydraminos_flag'] = contains_any(payload.get('liquor_flags', []), ['polyhydramnios','polihydramnios','polihydraminos'])
    payload['placenta_abruption_flag'] = contains_any(payload.get('current_pregnancy_menternal', []), ['placenta abruption','abruption'])
    payload['placenta_prev_or_abruption'] = contains_any(payload.get('current_pregnancy_menternal', []), ['placenta previa','placenta praevia','placenta abruption','abruption'])
    payload['severe_anemia'] = True if (payload.get('hb_g_dl') is not None and str(payload.get('hb_g_dl'))!='' and float(payload.get('hb_g_dl'))<7) else False
    payload['history_transfusion'] = contains_any(payload.get('menternal_medical', []), ['blood transfusion','transfusion'])
    payload['obstructed_prolonged'] = contains_any(payload.get('obstetric_history', []), ['obstructed','prolonged labor','prolonged'])

    # infer total_number_of_cs from obstetric_history when missing
    if payload.get('total_number_of_cs') in (None, '', np.nan):
        payload['total_number_of_cs'] = extract_cs_count_from_obstetric(payload.get('obstetric_history', []))

    # Coerce numeric where possible
    for k in NUMERIC_FEATURES:
        try:
            v = payload.get(k)
            if v in (None, ''):
                payload[k] = None
            else:
                payload[k] = float(v)
        except Exception:
            payload[k] = None

    # Apply deterministic rules first
    matches, total_weight, reasons = pph_deterministic_check(payload)
    print(matches)
    print(total_weight)
    print(reasons)
    if matches:
        # return detailed reasons and summed percent
        reason_details = matches
        total_pct = min(99, max(0, int(round(total_weight))))
        if total_pct >= rule_threshold:
            prob = total_pct / 100.0
            return {
                'prediction':'PPH',
                'probability': float(prob),
                'reason': '; '.join([m['reason'] for m in matches]),
                'reason_details': reason_details,
                'total_rule_percent': total_pct,
                'source':'rule'
            }
    else:
        return {
            'prediction':'NO_PPH',
            'probability': 0.0,
            'reason': '',
            'reason_details': matches,
            'total_rule_percent': min(99, max(0, int(round(total_weight)))),
            'source':'rule'
        }

    # require at least some data to run model
    has_any = any(payload.get(f) not in (None, '') for f in NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    if not has_any:
        return {'error':'Insufficient data for model prediction','source':'insufficient_data','status':400}

    # Load pipeline
    try:
        pipeline = load_pipeline()
    except FileNotFoundError as e:
        return {'error': str(e), 'source':'model_missing'}

    X_row = build_X_row_from_payload(payload)

    # predict
    # try:
    #     proba = None
    #     if hasattr(pipeline, 'predict_proba'):
    #         probs = pipeline.predict_proba(X_row)
    #         if probs.ndim == 2 and probs.shape[1] > 1:
    #             proba = float(probs[:,1][0])
    #     if proba is None:
    #         # fallback to predict label
    #         label = pipeline.predict(X_row)[0]
    #         proba = 1.0 if int(label) == 1 else 0.0
    #     pred = 'PPH' if proba >= threshold else 'NO_PPH'
    #     # reason: either rule suggestions (low weight) or model top features
    #     if matches and total_weight>0:
    #         reason = 'Rule signals: ' + '; '.join(reasons)
    #     else:
    #         try:
    #             clf = pipeline.named_steps.get('classifier', None)
    #             preproc = pipeline.named_steps.get('preprocessor', None)
    #             feat_names = None
    #             if preproc is not None:
    #                 try:
    #                     feat_names = list(preproc.get_feature_names_out())
    #                 except Exception:
    #                     # best-effort: build names from categorical OHE
    #                     try:
    #                         ohe = preproc.named_transformers_['cat'].named_steps['ohe']
    #                         cat_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    #                         feat_names = NUMERIC_FEATURES + list(cat_names) + BINARY_FEATURES
    #                     except Exception:
    #                         feat_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    #             else:
    #                 feat_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    #             importances = clf.feature_importances_ if clf is not None and hasattr(clf, 'feature_importances_') else None
    #             if importances is not None and feat_names is not None:
    #                 pairs = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)[:4]
    #                 reason = 'Top factors: ' + ', '.join([f"{n} ({imp:.3f})" for n,imp in pairs])
    #             else:
    #                 reason = 'Model-based prediction'
    #         except Exception:
    #             reason = 'Model-based prediction'
    #     return {'prediction': pred, 'probability': proba, 'reason': reason, 'source':'model', 'rules': matches}
    # except Exception as e:
    #     return {'error': f'Prediction failed: {str(e)}', 'source':'predict'}

# --- Identifier helper ---
def predict_pph_by_identifier(identifier, threshold=0.5, rule_threshold=5):
    # payload-like
    if isinstance(identifier, dict) or isinstance(identifier, list) or (isinstance(identifier, str) and identifier.strip().startswith('{')):
        return predict_pph_direct_payload(identifier, threshold=threshold, rule_threshold=rule_threshold)
    if Patient is None:
        return {'error':'Patient model not importable','source':'db'}
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
        return {'error': f'Error looking up Patient: {str(e)}', 'source':'db'}
    if patient is None:
        return {'error': f'Patient not found for identifier: {identifier}', 'status':404, 'source':'not_found'}
    payload = patient_to_payload(patient)
    return predict_pph_direct_payload(payload, threshold=threshold, rule_threshold=rule_threshold)

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
    # derive flags and coerce numbers
    payload['liquor_flags'] = [s.lower() for s in safe_parse_list(payload.get('liquor', [])) if isinstance(s, str)]
    payload['history_pph'] = contains_any(payload.get('obstetric_history', []), ['postpartum hemorrhage','pph'])
    payload['grand_multipara'] = contains_any(payload.get('social', []), ['grand multipara']) or (payload.get('parity') is not None and str(payload.get('parity'))!='' and float(payload.get('parity'))>=5)
    payload['multiple_gestation'] = contains_any(payload.get('current_pregnancy_menternal', []), ['multiple gestation','twin','twins','triplet'])
    payload['polihydraminos_flag'] = contains_any(payload.get('liquor_flags', []), ['polyhydramnios','polihydramnios','polihydraminos'])
    payload['placenta_abruption_flag'] = contains_any(payload.get('current_pregnancy_menternal', []), ['placenta abruption','abruption'])
    payload['placenta_prev_or_abruption'] = contains_any(payload.get('current_pregnancy_menternal', []), ['placenta previa','placenta praevia','placenta abruption','abruption'])
    payload['severe_anemia'] = True if (payload.get('hb_g_dl') is not None and str(payload.get('hb_g_dl'))!='' and float(payload.get('hb_g_dl'))<7) else False
    payload['history_transfusion'] = contains_any(payload.get('menternal_medical', []), ['blood transfusion','transfusion'])
    payload['obstructed_prolonged'] = contains_any(payload.get('obstetric_history', []), ['obstructed','prolonged labor','prolonged'])
    # coerce numeric
    for k in NUMERIC_FEATURES:
        try:
            if payload.get(k) is not None and str(payload.get(k))!='':
                payload[k] = float(payload[k])
            else:
                payload[k] = None
        except Exception:
            payload[k] = None
    # infer cs count
    if payload.get('total_number_of_cs') in (None, '', np.nan):
        payload['total_number_of_cs'] = extract_cs_count_from_obstetric(payload.get('obstetric_history', []))
    return payload
