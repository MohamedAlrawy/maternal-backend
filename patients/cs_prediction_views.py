
"""
File: views.py or api/views.py
CS Prediction API endpoint
"""

import os
import json
import pickle
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.core.exceptions import ObjectDoesNotExist
from patients.models import Patient
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny


# Direct rule-based predictions with percentages and reasons
DIRECT_PREDICTION_RULES = {
    # Obstetric history rules
    'cs_history_3plus': {
        'percentage': 99,
        'reason': 'Three or more prior CS nearly always delivered by CS for safety',
        'check': lambda p: check_in_json(p.obstetric_history, ['3 cs', '3+ cs', 'multiple cs', '4 cs', '5 cs', '(>3)'])
    },
    'cs_history_2': {
        'percentage': 90,
        'reason': 'Two prior CS usually lead to elective repeat CS due to rupture risk',
        'check': lambda p: check_in_json(p.obstetric_history, ['2 cs', 'two cs', 'c-sections (2)'])
    },
    'cs_history_1': {
        'percentage': 35,
        'reason': 'Trial of labor after one CS possible, but ~1/3 end with CS',
        'check': lambda p: check_in_json(p.obstetric_history, ['1 cs', 'one cs', 'c-section (1)'])
    },
    'uterine_rupture': {
        'percentage': 100,
        'reason': 'Previous uterine rupture mandates scheduled CS before labor',
        'check': lambda p: check_in_json(p.obstetric_history, ['uterine rupture', 'rupture', 'uterine rupture history'])
    },
    
    # Current pregnancy - maternal complications
    'placenta_previa_current': {
        'percentage': 99,
        'reason': 'Placenta covering cervix blocks vaginal delivery',
        'check': lambda p: check_in_json(p.current_pregnancy_menternal, ['placenta previa', 'previa'])
    },
    'placental_abruption_current': {
        'percentage': 85,
        'reason': 'Severe placental abruption often requires emergency CS to save mother and fetus',
        'check': lambda p: check_in_json(p.current_pregnancy_menternal, ['abruption', 'placental abruption'])
    },
    'multiple_gestation': {
        'percentage': 60,
        'reason': 'Twins or higher pregnancies have increased CS risk, especially if malpresentation',
        'check': lambda p: check_in_json(p.current_pregnancy_menternal, ['multiple gestation', 'twins', 'triplet'])
    },
    'preeclampsia_current': {
        'percentage': 60,
        'reason': 'Severe preeclampsia often requires CS for maternal/fetal safety',
        'check': lambda p: check_in_json(p.current_pregnancy_menternal, ['preeclampsia', 'pre-eclampsia'])
    },
    'ivf_icsi': {
        'percentage': 85,
        'reason': 'IVF/ICSI pregnancies show higher elective/emergency CS rates due to obstetric risks and "precious baby" effect',
        'check': lambda p: check_in_json(p.current_pregnancy_menternal, ['ivf', 'icsi'])
    },
    'severe_anemia': {
        'percentage': 25,
        'reason': 'Severe anemia (Hb<7) limits labor tolerance; CS often chosen if complications exist',
        'check': lambda p: check_in_json(p.current_pregnancy_menternal, ['severe anemia']) and (p.hb_g_dl or 8) >= 7
    },
    
    # Current pregnancy - fetal factors
    'non_cephalic_presentation': {
        'percentage': 90,
        'reason': 'Breech/transverse lies usually managed with CS to reduce perinatal risk',
        'check': lambda p: check_in_json(p.current_pregnancy_fetal, ['breech', 'transverse', 'oblique']) or 
                          p.presentation in ['preech', 'transverse', 'oblique']
    },
    'estimated_fetal_weight_4000': {
        'percentage': 50,
        'reason': 'Large babies (≥4000g) linked to CPD, shoulder dystocia, higher CS rate',
        'check': lambda p: (p.estimated_fetal_weight_by_gm or 0) >= 4000
    },
    
    # Intrapartum factors
    'ctg_category_iii': {
        'percentage': 95,
        'reason': 'Category III CTG = urgent CS for suspected hypoxia/acidosis',
        'check': lambda p: p.ctg_category == 'category_iii_pathological'
    },
    'ctg_category_ii': {
        'percentage': 70,
        'reason': 'Category II CTG often monitored but may require CS if unresolved',
        'check': lambda p: p.ctg_category == 'category_ii_suspicious'
    },
}


def check_in_json(json_list, keywords):
    """Check if any keyword exists in JSON list (case-insensitive)"""
    if not json_list:
        return False
    
    json_str = str(json_list).lower()
    for keyword in keywords:
        if keyword.lower() in json_str:
            return True
    return False


def load_ml_model():
    """Load trained ML model and scaler"""
    model_dir = 'ml_models'
    
    try:
        model_path = os.path.join(model_dir, 'cs_prediction_model.pkl')
        scaler_path = os.path.join(model_dir, 'cs_prediction_scaler.pkl')
        features_path = os.path.join(model_dir, 'cs_prediction_features.json')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(features_path, 'r') as f:
            feature_names = json.load(f)
        
        return model, scaler, feature_names
    except Exception as e:
        return None, None, None


def extract_features_from_patient(patient):
    """Extract features from patient record for ML prediction"""
    features = {}
    
    # Basic demographics
    features['age'] = patient.age or 0
    features['parity'] = patient.parity or 0
    features['gravidity'] = patient.gravidity or 0
    features['bmi'] = float(patient.bmi) if patient.bmi else 0
    
    # Vital signs
    features['pulse'] = patient.pulse or 0
    features['temp'] = float(patient.temp) if patient.temp else 0
    features['oxygen_sat'] = patient.oxygen_sat or 0
    
    # Blood pressure
    try:
        bp_parts = patient.bp.split('/') if patient.bp else ['0', '0']
        features['systolic_bp'] = int(bp_parts[0]) if len(bp_parts) > 0 else 0
        features['diastolic_bp'] = int(bp_parts[1]) if len(bp_parts) > 1 else 0
    except:
        features['systolic_bp'] = 0
        features['diastolic_bp'] = 0
    
    # Fetal weight
    features['estimated_fetal_weight'] = patient.estimated_fetal_weight_by_gm or 0
    
    # Labor duration
    features['labor_duration'] = patient.labor_duration_hours or 0
    
    # Cervical dilation at admission
    features['cervical_dilation_admission'] = patient.cervical_dilatation_at_admission or 0
    
    # Binary flags for medical history
    features['chronic_hypertension'] = check_in_json(
        patient.menternal_medical, ['chronic hypertension', 'hypertension']
    )
    features['diabetes'] = check_in_json(
        patient.menternal_medical, ['diabetes', 'gdm', 'gestational diabetes']
    )
    features['preeclampsia_history'] = check_in_json(
        patient.obstetric_history, ['preeclampsia', 'pre-eclampsia']
    )
    features['cs_history_1'] = check_in_json(
        patient.obstetric_history, ['previous c-section', '1 cs', 'one cs', 'c-section (1)']
    )
    features['cs_history_2'] = check_in_json(
        patient.obstetric_history, ['2 cs', 'two cs', 'c-sections (2)']
    )
    features['cs_history_3plus'] = check_in_json(
        patient.obstetric_history, ['3 cs', '3+ cs', 'multiple cs', '4 cs', '5 cs', '(>3)']
    )
    features['uterine_rupture_history'] = check_in_json(
        patient.obstetric_history, ['uterine rupture', 'rupture']
    )
    features['cardiac_disease'] = check_in_json(
        patient.menternal_medical, ['cardiac', 'heart disease']
    )
    features['hiv_immunocompromised'] = check_in_json(
        patient.menternal_medical, ['hiv', 'immunocompromised']
    )
    features['grand_multipara'] = check_in_json(
        patient.social, ['grand multipara', '>=5', '>5']
    )
    
    # Current pregnancy factors
    features['multiple_gestation'] = check_in_json(
        patient.current_pregnancy_menternal, ['multiple gestation', 'twins', 'triplet']
    )
    features['placenta_previa'] = check_in_json(
        patient.current_pregnancy_menternal, ['placenta previa', 'previa']
    )
    features['placental_abruption'] = check_in_json(
        patient.current_pregnancy_menternal, ['abruption', 'placental abruption']
    )
    features['severe_anemia'] = check_in_json(
        patient.current_pregnancy_menternal, ['severe anemia', 'anemia']
    )
    features['ivf_icsi'] = check_in_json(
        patient.current_pregnancy_menternal, ['ivf', 'icsi']
    )
    
    # Fetal factors
    features['non_cephalic_presentation'] = check_in_json(
        patient.current_pregnancy_fetal, ['breech', 'transverse', 'oblique', 'non-cephalic']
    )
    features['iugr'] = check_in_json(
        patient.current_pregnancy_fetal, ['iugr', 'growth restriction']
    )
    
    # Presentation and delivery factors
    is_non_cephalic = patient.presentation in ['preech', 'transverse', 'oblique']
    features['presentation_non_cephalic'] = 1 if is_non_cephalic else 0
    features['multiple_fetuses'] = 1 if patient.fetus_number in ['twin', 'triplete'] else 0
    
    # CTG category
    features['ctg_category_ii'] = 1 if patient.ctg_category == 'category_ii_suspicious' else 0
    features['ctg_category_iii'] = 1 if patient.ctg_category == 'category_iii_pathological' else 0
    
    # Bishop score / cervix favorability
    features['cervix_unfavorable'] = 1 if (
        patient.cervix_favrable_for_induction == 'unfavorable_bishop_score_less_6'
    ) else 0
    
    # Hemoglobin
    features['hb'] = patient.hb_g_dl or 0
    
    # BMI categories
    features['bmi_35_39_5'] = 1 if (35 <= features['bmi'] < 39.5) else 0
    features['bmi_40_plus'] = 1 if features['bmi'] >= 40 else 0
    
    return features


@api_view(['GET'])
@permission_classes([AllowAny])
def predict_patient_by_identifier(request):
    """
    API endpoint to predict CS probability for a patient
    
    Query params:
    - patient_id: Patient ID
    - OR file_number: File number
    
    Returns:
    {
        'success': bool,
        'cs_probability': float (0-100),
        'prediction_method': 'direct_rule' or 'ml_model',
        'reason': str,
        'confidence': str (high/medium/low),
        'risk_factors': list
    }
    """
    try:
        # Get patient identifier
        patient_id = request.query_params.get('patient_id')
        file_number = request.query_params.get('file_number')
        
        if not patient_id and not file_number:
            return Response({
                'success': False,
                'error': 'Please provide either patient_id or file_number'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Fetch patient
        try:
            if patient_id:
                patient = Patient.objects.get(patient_id=patient_id)
            else:
                patient = Patient.objects.get(file_number=file_number)
        except ObjectDoesNotExist:
            return Response({
                'success': False,
                'error': 'Patient not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # Step 1: Check direct prediction rules
        direct_prediction = check_direct_rules(patient)
        if direct_prediction:
            return Response({
                'success': True,
                'patient_id': patient.patient_id,
                'file_number': patient.file_number,
                'patient_name': patient.name,
                'cs_probability': direct_prediction['percentage'],
                'prediction_method': 'direct_rule',
                'reason': direct_prediction['reason'],
                'confidence': 'high',
                'risk_factors': direct_prediction.get('risk_factors', [])
            })
        
        # Step 2: Use ML model if available
        model, scaler, feature_names = load_ml_model()
        
        if model is None:
            return Response({
                'success': False,
                'error': 'ML model not trained yet. Run: python manage.py train_cs_prediction'
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        
        # Extract features
        features = extract_features_from_patient(patient)
        
        # Check if all required features are available
        missing_features = [f for f in feature_names if f not in features]
        if missing_features:
            return Response({
                'success': False,
                'error': f'Insufficient data for prediction. Missing: {", ".join(missing_features)}'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Prepare feature vector
        X = np.array([[features[f] for f in feature_names]])
        X_scaled = scaler.transform(X)
        
        # Make prediction
        cs_probability = float(model.predict_proba(X_scaled)[0][1]) * 100
        confidence = get_confidence_level(cs_probability, model)
        
        # Identify contributing risk factors and generate reason
        feature_importance = model.feature_importances_
        top_features = sorted(
            zip(feature_names, feature_importance),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        active_risk_factors = []
        top_active_factors = []
        
        for fname, importance in top_features:
            if features[fname]:  # If feature is present/True
                active_risk_factors.append(f"{fname}: {importance:.3f}")
                top_active_factors.append((fname, importance))
        
        # Generate reason based on active risk factors
        reason = generate_ml_reason(top_active_factors, cs_probability)
        
        return Response({
            'success': True,
            'patient_id': patient.patient_id,
            'file_number': patient.file_number,
            'patient_name': patient.name,
            'cs_probability': round(cs_probability, 2),
            'prediction_method': 'ml_model',
            'reason': reason,
            'confidence': confidence,
            'risk_factors': active_risk_factors,
            'note': 'Prediction based on machine learning model trained on historical data'
        })
    
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def check_direct_rules(patient):
    """
    Check if patient matches any direct prediction rules
    Returns highest matching rule or None
    """
    matching_rules = []
    
    for rule_key, rule in DIRECT_PREDICTION_RULES.items():
        print(rule_key, "dddddddddddddddd", rule)
        print(rule['check'](patient))
        try:
            if rule['check'](patient):
                matching_rules.append({
                    'rule': rule_key,
                    'percentage': rule['percentage'],
                    'reason': rule['reason'],
                    'risk_factors': [rule_key]
                })
        except:
            continue
    
    if not matching_rules:
        return None
    
    # Return the rule with highest CS probability
    highest_rule = max(matching_rules, key=lambda x: x['percentage'])
    return highest_rule


def get_confidence_level(probability, model):
    """Determine confidence level based on probability and model characteristics"""
    if probability >= 80:
        return 'high'
    elif probability >= 50:
        return 'medium'
    else:
        return 'low'


# Map feature names to human-readable clinical reasons
FEATURE_REASON_MAP = {
    'cs_history_3plus': 'Multiple prior cesarean sections (3+) - repeated CS increases risk',
    'cs_history_2': 'Two prior cesarean sections - increases repeat CS likelihood',
    'cs_history_1': 'One prior cesarean section - trial of labor possible but ~1/3 result in CS',
    'uterine_rupture_history': 'History of uterine rupture - mandates cesarean for safety',
    'placenta_previa': 'Placenta previa detected - blocks vaginal delivery route',
    'placental_abruption': 'Placental abruption - requires emergency intervention',
    'multiple_gestation': 'Multiple pregnancy (twins/triplets) - increased CS risk',
    'preeclampsia_history': 'History of preeclampsia - elevated maternal/fetal risk',
    'non_cephalic_presentation': 'Non-cephalic presentation (breech/transverse) - requires CS',
    'estimated_fetal_weight': 'Large estimated fetal weight - risk of cephalopelvic disproportion',
    'ctg_category_iii': 'Category III CTG (pathological) - urgent CS needed for fetal distress',
    'ctg_category_ii': 'Category II CTG (suspicious) - increased monitoring and CS risk',
    'chronic_hypertension': 'Chronic hypertension - increases abruption and fetal compromise risk',
    'diabetes': 'Diabetes mellitus - increases gestational complications and CS risk',
    'cardiac_disease': 'Cardiac disease - pregnancy complications may necessitate CS',
    'hiv_immunocompromised': 'HIV/immunocompromised status - may require planned CS',
    'ivf_icsi': 'Pregnancy via IVF/ICSI - higher elective and emergency CS rates',
    'severe_anemia': 'Severe anemia - limits labor tolerance, lower CS threshold',
    'grand_multipara': 'Grand multiparity (≥5 births) - mixed risk factors',
    'labor_duration': 'Extended labor duration - may necessitate intervention',
    'cervical_dilation_admission': 'Cervical dilation at admission - affects labor progression',
    'cervix_unfavorable': 'Unfavorable cervix for labor - reduced induction success',
    'presentation_non_cephalic': 'Non-cephalic presentation documented - requires cesarean',
    'multiple_fetuses': 'Multiple fetuses confirmed - higher CS risk',
    'age': 'Advanced maternal age - increases obstetric complications',
    'parity': 'Parity status - affects labor outcomes and CS risk',
    'bmi': 'Body Mass Index - obesity increases CS risk and complications',
    'bmi_35_39_5': 'Overweight (BMI 35-39.5) - increased CS risk',
    'bmi_40_plus': 'Obesity (BMI ≥40) - significantly increased CS risk',
}


def generate_ml_reason(top_active_factors, probability):
    """
    Generate clinical reason for ML model prediction
    
    Args:
        top_active_factors: List of tuples (feature_name, importance_score)
        probability: CS probability percentage
    
    Returns:
        String explaining the prediction reason
    """
    if not top_active_factors:
        if probability >= 80:
            return 'Multiple risk factors identified by model - high CS likelihood'
        elif probability >= 50:
            return 'Moderate risk factors identified - close monitoring recommended'
        else:
            return 'Low risk profile identified - likely vaginal delivery candidate'
    
    # Build reason from top contributing factors
    factors_reasons = []
    for feature_name, importance in top_active_factors[:3]:  # Top 3 factors
        if feature_name in FEATURE_REASON_MAP:
            factors_reasons.append(FEATURE_REASON_MAP[feature_name])
    
    if not factors_reasons:
        factors_reasons.append('Multiple clinical factors favor cesarean delivery')
    
    # Create compound reason
    if len(factors_reasons) == 1:
        reason = factors_reasons[0]
    elif len(factors_reasons) == 2:
        reason = f"{factors_reasons[0]}; {factors_reasons[1]}"
    else:
        reason = f"{factors_reasons[0]}; {factors_reasons[1]}; and {factors_reasons[2]}"
    
    # Add confidence statement based on probability
    if probability >= 80:
        reason += ' - High confidence prediction'
    elif probability >= 60:
        reason += ' - Moderate confidence prediction'
    else:
        reason += ' - Lower confidence prediction'
    
    return reason


# Additional endpoint to get prediction details and interpretation
@api_view(['GET'])
def get_cs_prediction_info(request):
    """
    Get detailed information about CS prediction
    Includes model statistics and risk factor explanations
    """
    model, scaler, feature_names = load_ml_model()
    
    if model is None:
        return Response({
            'success': False,
            'error': 'ML model not available'
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    
    # Get feature importance
    feature_importance = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Get direct rules info
    direct_rules_info = []
    for rule_key, rule in DIRECT_PREDICTION_RULES.items():
        direct_rules_info.append({
            'rule': rule_key,
            'percentage': rule['percentage'],
            'reason': rule['reason']
        })
    
    return Response({
        'success': True,
        'model_type': 'Random Forest Classifier',
        'feature_importance': [
            {'feature': f[0], 'importance': float(f[1])}
            for f in feature_importance[:15]
        ],
        'direct_rules': direct_rules_info,
        'total_features': len(feature_names),
        'all_features': feature_names
    })
