"""
Neonatal predictor (direct feature set) - strict for missing direct fields

Behavior:
- Applies deterministic rules first (high-confidence short-circuit).
- If rules don't decide, requires a minimal set of direct fields to be present (numeric or categorical).
  If required direct fields are missing -> returns error with missing_fields and status 400.
- Otherwise uses trained pipeline (artifacts/neonatal_direct_pipeline.joblib) to predict.
Place as: patients/neonatal_predictor_direct.py
"""
import os, json, re, joblib, numpy as np, pandas as pd
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist

try:
    from .models import Patient
except Exception:
    Patient = None

MODEL_FILENAME = 'neonatal_direct_pipeline.joblib'
MODEL_PATH = os.path.join(getattr(settings,'BASE_DIR','.'), 'artifacts', MODEL_FILENAME)

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

# Deterministic rules (same weights as training command)
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

# Feature lists (direct set)
NUMERIC_FEATURES = ['age','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','labor_duration_hours','apgar_score']
CATEGORICAL_FEATURES = ['ctg_category','mode_of_delivery','instrumental_delivery','membrane_status','rupture_duration_hour','fetus_number','placenta_location']
BINARY_FEATURES = ['preterm_birth_less_37_weeks','placenta_prev_or_abruption','multiple_gestation','non_cephalic','iugr_flag','ivf_flag','history_still_neonatal_death','congenital_anomalies','grand_multipara','severe_anemia','chronic_hypertension','diabetes_any','pre_eclampsia']

def load_pipeline(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)

def patient_to_payload(patient):
    payload = {}
    simple_fields = ['id','age','parity','mode_of_delivery','instrumental_delivery','labor_duration_hours','rupture_duration_hour','membrane_status','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','ctg_category','fetus_number','congenital_anomalies','neonatal_death','nicu_admission','hie','birth_injuries','apgar_score','preterm_birth_less_37_weeks','total_number_of_cs','parity','placenta_location']
    for f in simple_fields:
        try:
            v = getattr(patient, f)
            payload[f] = None if v is None else (v.isoformat() if hasattr(v,'isoformat') else v)
        except Exception:
            payload[f] = None
    json_list_cols = ['menternal_medical','obstetric_history','current_pregnancy_menternal','current_pregnancy_fetal','social','liquor']
    for col in json_list_cols:
        raw = getattr(patient, col, None)
        payload[col] = safe_parse_list(raw)
    # coerce numeric where possible
    for k in NUMERIC_FEATURES + ['total_number_of_cs','parity']:
        try:
            if payload.get(k) is not None:
                payload[k] = float(payload[k])
        except Exception:
            payload[k] = None
    payload['liquor_flags'] = [s.lower() for s in safe_parse_list(payload.get('liquor', [])) if isinstance(s,str)]
    # derived flags
    payload['placenta_abruption_flag'] = contains_any(payload.get('current_pregnancy_menternal', []), ['placenta abruption','abruption'])
    payload['placenta_prev_or_abruption'] = contains_any(payload.get('current_pregnancy_menternal', []), ['placenta previa','placenta praevia','placenta abruption','abruption'])
    payload['multiple_gestation'] = contains_any(payload.get('current_pregnancy_menternal', []), ['multiple gestation','twin','twins','triplet'])
    payload['non_cephalic'] = contains_any(payload.get('current_pregnancy_fetal', []), ['breech','non-cephalic','transverse','oblique'])
    payload['iugr_flag'] = contains_any(payload.get('current_pregnancy_fetal', []), ['iugr','intrauterine growth restriction','sga'])
    payload['ivf_flag'] = contains_any(payload.get('current_pregnancy_menternal', []), ['pregnancy post ivf','ivf','icsi'])
    payload['polihydraminos_flag'] = contains_any(payload.get('liquor_flags', []), ['polyhydramnios','polihydramnios','polihydraminos'])
    payload['history_still_neonatal_death'] = contains_any(payload.get('obstetric_history', []), ['stillbirth','neonatal death'])
    payload['preterm_birth_less_37_weeks'] = True if (str(payload.get('preterm_birth_less_37_weeks','')).lower() in ['true','1','yes'] or payload.get('preterm_birth_less_37_weeks') is True) else False
    payload['severe_anemia'] = True if (payload.get('hb_g_dl') is not None and payload.get('hb_g_dl')<7) else False
    return payload

def predict_neonatal_direct_payload(payload, threshold=0.5, rule_threshold=80):
    try:
        # first apply deterministic rules
        matched, weight, reasons = neonatal_deterministic_check(payload)
        if matched and weight >= rule_threshold:
            return {'prediction':'ADVERSE', 'probability': min(0.99, weight/100.0), 'reason': '; '.join(reasons), 'source':'rule', 'adverse_neonatal': True}
        # require presence of at least one numeric and one categorical field from the direct set
        missing_fields = []
        has_numeric = any(payload.get(k) not in (None,'') for k in NUMERIC_FEATURES)
        has_categorical = any(payload.get(k) not in (None,'') for k in CATEGORICAL_FEATURES)
        if not has_numeric or not has_categorical:
            # list missing fields specifically
            for k in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
                if payload.get(k) in (None,''):
                    missing_fields.append(k)
            return {'error':'Missing required direct fields for neonatal prediction', 'missing_fields': missing_fields, 'status':400, 'source':'insufficient_data'}
        # load model
        try:
            pipeline = load_pipeline()
        except FileNotFoundError as fnf:
            return {'error': str(fnf), 'source': 'model_missing'}
        # build X_row
        rec = {}
        for k in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
            rec[k] = payload.get(k, None)
        X_row = pd.DataFrame([rec], columns=(NUMERIC_FEATURES + CATEGORICAL_FEATURES))
        # add binary features as 0/1
        for b in BINARY_FEATURES:
            val = payload.get(b, False)
            X_row[b] = 1 if (str(val).lower() in ['true','1','yes'] or val is True or val==1) else 0
        feature_order = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
        X_row = X_row[feature_order]
        # predict
        try:
            probs = None
            if hasattr(pipeline, 'predict_proba'):
                probs = pipeline.predict_proba(X_row)
            if probs is not None and probs.ndim==2 and probs.shape[1]>1:
                proba = float(probs[:,1][0])
                pred = 'ADVERSE' if proba >= threshold else 'OK'
            else:
                pred_label = pipeline.predict(X_row)[0]
                proba = None
                pred = 'ADVERSE' if int(pred_label)==1 else 'OK'
            # compute reason: include deterministic reasons if present but below threshold
            if matched and weight>0 and weight < rule_threshold:
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
            return {'prediction': pred, 'probability': proba, 'reason': reason, 'source':'model', 'adverse_neonatal': True if pred=='ADVERSE' else False}
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}', 'source':'predict'}
    except Exception as e:
        return {'error': f'Neonatal direct prediction exception: {str(e)}', 'source':'predict'}

def predict_neonatal_by_identifier(identifier, threshold=0.5, rule_threshold=80):
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
    return predict_neonatal_direct_payload(payload, threshold=threshold, rule_threshold=rule_threshold)
