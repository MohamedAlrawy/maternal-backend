"""
Neonatal predictor module: patients/neonatal_predictor_refactored.py

Provides:
 - deterministic-rule-first inference
 - ML fallback using artifacts/neonatal_pipeline.joblib
 - patient_to_payload and predict_neonatal_by_identifier utilities
 - returns 400-style error when required numeric fields missing
"""

import os, json, re, joblib, numpy as np, pandas as pd
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist

try:
    from .models import Patient
except Exception:
    Patient = None

MODEL_FILENAME = 'neonatal_pipeline.joblib'
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

NEO_RULES = [
    (lambda r: r.get('preterm_birth_less_37_weeks', False), 50, "Preterm birth <37 weeks"),
    (lambda r: (str(r.get('ctg_category','')).lower().find('category_iii')>=0 or str(r.get('ctg_category','')).lower().find('category iii')>=0), 85, "CTG Category III"),
    (lambda r: (str(r.get('ctg_category','')).lower().find('category_ii')>=0 or str(r.get('ctg_category','')).lower().find('category ii')>=0), 30, "CTG Category II"),
    (lambda r: r.get('placenta_abruption_flag', False), 60, "Placental abruption"),
    (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['placenta previa','placenta praevia','placenta abruption','abruption']), 25, "Placenta previa/abruption"),
    (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['multiple gestation','twin','twins','triplet']), 25, "Multiple gestation"),
    (lambda r: contains_any(r.get('current_pregnancy_fetal', []), ['breech','non-cephalic','transverse','oblique']), 15, "Non-cephalic presentation"),
    (lambda r: (str(r.get('rupture_duration_hour','')).lower().find('18')>=0 or str(r.get('rupture_duration_hour','')).lower().find('24')>=0), 12, "Prolonged rupture (18-24h)"),
    (lambda r: contains_any(r.get('indication_of_induction', []), ['prelabor_rupture_of_membranes_prom','prelabor rupture','prom']), 8, "PROM"),
    (lambda r: contains_any(r.get('obstetric_history', []), ['preeclampsia','pre-eclampsia','pre eclampsia']), 20, "History of preeclampsia"),
    (lambda r: contains_any(r.get('menternal_medical', []), ['chronic hypertension','hypertension']), 12, "Chronic hypertension"),
    (lambda r: contains_any(r.get('menternal_medical', []), ['diabetes','gdm','gestational diabetes']), 12, "Diabetes"),
    (lambda r: (not pd.isna(r.get('hb_g_dl')) and float(r.get('hb_g_dl',999)) < 7), 8, "Severe maternal anemia (Hb<7)"),
    (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['pregnancy post ivf','ivf','icsi']), 12, "Pregnancy post IVF/ICSI"),
    (lambda r: contains_any(r.get('current_pregnancy_fetal', []), ['iugr','intrauterine growth restriction','sga']), 20, "IUGR"),
    (lambda r: contains_any(r.get('obstetric_history', []), ['stillbirth','neonatal death']), 30, "Prior stillbirth/neonatal death"),
    (lambda r: contains_any(r.get('current_pregnancy_menternal', []), ['polihydraminos','polyhydramnios','polihydramnios']), 12, "Polyhydramnios"),
    (lambda r: (r.get('total_number_of_cs',0) > 1), 8, "Multiple prior CS (>1)"),
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

def load_pipeline(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)

def build_feature_row(payload, required_numeric=None, required_categorical=None):
    if required_numeric is None:
        required_numeric = ['age','labor_duration_hours','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','apgar_score']
    if required_categorical is None:
        required_categorical = ['mode_of_delivery','instrumental_delivery','rupture_duration_hour','membrane_status','ctg_category','fetus_number']
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

def predict_neonatal_payload(payload: dict, threshold=0.5, rule_threshold=80):
    try:
        list_fields = ['menternal_medical','obstetric_history','current_pregnancy_menternal','current_pregnancy_fetal','social','liquor']
        for lf in list_fields:
            payload[lf] = safe_parse_list(payload.get(lf, []))
        payload['liquor_flags'] = [s.lower() for s in safe_parse_list(payload.get('liquor', [])) if isinstance(s, str)]
        payload['preterm_birth_less_37_weeks'] = True if (str(payload.get('preterm_birth_less_37_weeks','')).lower() in ['true','1','yes'] or payload.get('preterm_birth_less_37_weeks') is True) else False
        payload['placenta_abruption_flag'] = contains_any(payload.get('current_pregnancy_menternal', []), ['placenta abruption','abruption'])
        payload['multiple_gestation'] = contains_any(payload.get('current_pregnancy_menternal', []), ['multiple gestation','twin','twins','triplet'])
        payload['non_cephalic'] = contains_any(payload.get('current_pregnancy_fetal', []), ['breech','non-cephalic','transverse','oblique'])
        payload['iugr_flag'] = contains_any(payload.get('current_pregnancy_fetal', []), ['iugr','intrauterine growth restriction','sga'])
        payload['ivf_flag'] = contains_any(payload.get('current_pregnancy_menternal', []), ['pregnancy post ivf','ivf','icsi'])
        payload['polihydraminos_flag'] = contains_any(payload.get('liquor_flags', []), ['polyhydramnios','polihydramnios','polihydraminos'])
        payload['history_still_neonatal_death'] = contains_any(payload.get('obstetric_history', []), ['stillbirth','neonatal death'])
        for k in ['age','labor_duration_hours','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','apgar_score','total_number_of_cs']:
            if k in payload:
                try:
                    payload[k] = float(payload[k]) if payload[k] not in (None,'') else None
                except Exception:
                    payload[k] = None
        payload['total_number_of_cs_gt1'] = 1 if (payload.get('total_number_of_cs') is not None and payload.get('total_number_of_cs')>1) else 0

        matched, weight, reasons = neonatal_deterministic_check(payload)
        if matched and weight >= rule_threshold:
            return {'prediction': 'ADVERSE', 'probability': min(0.99, weight/100.0), 'reason': '; '.join(reasons), 'source': 'rule', 'adverse_neonatal': True}

        try:
            pipeline = load_pipeline()
        except FileNotFoundError as fnf:
            return {'error': str(fnf), 'source': 'model_missing'}

        numeric_features = ['age','labor_duration_hours','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','apgar_score']
        categorical_features = ['mode_of_delivery','instrumental_delivery','rupture_duration_hour','membrane_status','ctg_category','fetus_number']
        binary_features = ['chronic_hypertension','diabetes_any','pre_eclampsia','severe_anemia','total_number_of_cs_gt1','grand_multipara','preterm_birth_less_37_weeks','placenta_prev_or_abruption','polihydraminos_flag','multiple_gestation','non_cephalic','iugr_flag','ivf_flag','history_still_neonatal_death','congenital_anomalies']

        X_row, missing = build_feature_row(payload, required_numeric=numeric_features, required_categorical=categorical_features)
        if missing:
            return {'error': 'Missing required numeric fields for neonatal prediction', 'missing_fields': missing, 'status': 400, 'source': 'insufficient_data'}

        for b in binary_features:
            val = payload.get(b, False)
            X_row[b] = 1 if (str(val).lower() in ['true','1','yes'] or val is True or val==1) else 0

        feature_order = numeric_features + categorical_features + binary_features
        X_row = X_row[feature_order]
        try:
            proba = float(pipeline.predict_proba(X_row)[:,1][0])
            pred = 'ADVERSE' if proba >= threshold else 'OK'
            adverse_flag = True if proba >= threshold else False
            reason = None
            if matched and weight>0:
                reason = '; '.join(reasons)
            else:
                try:
                    clf = pipeline.named_steps['classifier']
                    preproc = pipeline.named_steps['preprocessor']
                    try:
                        cat_names = list(preproc.get_feature_names_out())
                    except Exception:
                        ohe = preproc.named_transformers_['cat'].named_steps['ohe']
                        cat_names = list(ohe.get_feature_names_out(categorical_features))
                    feature_names = numeric_features + list(cat_names) + binary_features
                    importances = clf.feature_importances_
                    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
                    top_features = [f"{name} ({imp:.3f})" for name, imp in feat_imp[:4]]
                    reason = f"Top influencing factors: {', '.join(top_features)}"
                except Exception:
                    reason = "Model-based prediction"
            return {'prediction': pred, 'probability': proba, 'reason': reason, 'source': 'model', 'adverse_neonatal': adverse_flag}
        except Exception as e:
            return {'error': f'Prediction failure: {str(e)}', 'source': 'predict'}
    except Exception as e:
        return {'error': f'Neonatal prediction exception: {str(e)}', 'source': 'predict'}

def patient_to_payload(patient):
    payload = {}
    simple_fields = ['id','age','mode_of_delivery','instrumental_delivery','labor_duration_hours','rupture_duration_hour','indication_of_induction','membrane_status','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','ctg_category','fetus_number','congenital_anomalies','neonatal_death','nicu_admission','hie','birth_injuries','apgar_score','preterm_birth_less_37_weeks','total_number_of_cs']
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
    json_list_cols = ['menternal_medical','obstetric_history','current_pregnancy_menternal','current_pregnancy_fetal','social','liquor']
    for col in json_list_cols:
        raw = getattr(patient, col, None)
        payload[col] = safe_parse_list(raw)
    for k in ['age','labor_duration_hours','estimated_fetal_weight_by_gm','hb_g_dl','platelets_x10e9l','apgar_score','total_number_of_cs']:
        try:
            if payload.get(k) is not None:
                payload[k] = float(payload[k])
        except Exception:
            payload[k] = None
    return payload

def predict_neonatal_by_identifier(identifier, threshold=0.5, rule_threshold=80):
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
    result = predict_neonatal_payload(payload, threshold=threshold, rule_threshold=rule_threshold)
    if isinstance(result, dict) and 'error' not in result:
        result['patient_pk'] = patient.pk
        result['file_number'] = getattr(patient, 'file_number', None)
    return result
