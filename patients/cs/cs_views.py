"""
Cesarean Section Prediction API
Function: predict_patient_by_identifier(patient_identifier)
"""
import json
import numpy as np
import joblib
from pathlib import Path
from django.db import models

# Import your Patient model
from patients.models import Patient


class CSPredictor:
    """Cesarean Section Prediction Engine"""
    
    # Direct rule definitions
    DIRECT_RULES = [
        {
            'field': 'obstetric_history',
            'condition': 'multiple_cs_gt3',
            'keywords': ['c-section', 'cs', 'cesarean', 'caesarean'],
            'min_count': 3,
            'percentage': 99,
            'reason': 'Three or more CS nearly always delivered by CS for safety',
            'confidence': 'high'
        },
        {
            'field': 'obstetric_history',
            'condition': 'multiple_cs_2',
            'keywords': ['c-section', 'cs', 'cesarean', 'caesarean'],
            'count': 2,
            'percentage': 90,
            'reason': 'Two prior CS usually lead to elective repeat CS due to rupture risk',
            'confidence': 'high'
        },
        {
            'field': 'obstetric_history',
            'condition': 'previous_cs_1',
            'keywords': ['c-section', 'cs', 'cesarean', 'caesarean'],
            'count': 1,
            'percentage': 35,
            'reason': 'Trial of labor after one CS possible, but ~1/3 end with CS',
            'confidence': 'moderate'
        },
        {
            'field': 'current_pregnancy_menternal',
            'condition': 'placenta_abruption',
            'keywords': ['placenta abruption', 'abruption', 'abruptio'],
            'percentage': 85,
            'reason': 'Severe abruption often requires emergency CS to save mother and fetus',
            'confidence': 'high'
        },
        {
            'field': 'obstetric_history',
            'condition': 'history_placenta_abruption',
            'keywords': ['placenta abruption', 'abruption', 'abruptio'],
            'percentage': 85,
            'reason': 'Severe abruption often requires emergency CS to save mother and fetus',
            'confidence': 'high'
        },
        {
            'field': 'current_pregnancy_menternal',
            'condition': 'placenta_previa',
            'keywords': ['placenta previa', 'previa', 'placenta praevia'],
            'percentage': 99,
            'reason': 'Placenta covering cervix blocks vaginal delivery',
            'confidence': 'high'
        },
        {
            'field': 'obstetric_history',
            'condition': 'history_placenta_previa',
            'keywords': ['placenta previa', 'previa', 'placenta praevia'],
            'percentage': 99,
            'reason': 'Placenta covering cervix blocks vaginal delivery',
            'confidence': 'high'
        },
        {
            'field': 'current_pregnancy_fetal',
            'condition': 'non_cephalic',
            'keywords': ['breech', 'transverse', 'oblique', 'malpresentation'],
            'percentage': 90,
            'reason': 'Breech/transverse lies usually managed with CS to reduce perinatal risk',
            'confidence': 'high'
        },
        {
            'field': 'current_pregnancy_menternal',
            'condition': 'multiple_gestation',
            'keywords': ['twins', 'twin', 'triplets', 'triplet', 'multiple gestation'],
            'percentage': 60,
            'reason': 'Twins or higher pregnancies have increased CS risk, especially if malpresentation',
            'confidence': 'moderate'
        },
        {
            'field': 'menternal_medical',
            'condition': 'chronic_hypertension',
            'keywords': ['chronic hypertension', 'hypertension', 'htn'],
            'percentage': 50,
            'reason': 'Chronic hypertension increases risk of abruption, fetal compromise → higher CS',
            'confidence': 'moderate'
        },
        {
            'field': 'current_pregnancy_menternal',
            'condition': 'preeclampsia',
            'keywords': ['pre-eclampsia', 'preeclampsia', 'eclampsia'],
            'percentage': 60,
            'reason': 'Severe preeclampsia often requires CS for maternal/fetal safety',
            'confidence': 'moderate'
        },
        {
            'field': 'current_pregnancy_menternal',
            'condition': 'severe_anemia',
            'keywords': ['severe anemia', 'anemia', 'anaemia'],
            'percentage': 25,
            'reason': 'Severe anemia limits tolerance for labor; CS often chosen if complications exist',
            'confidence': 'low'
        },
        {
            'field': 'menternal_medical',
            'condition': 'cardiac_disease',
            'keywords': ['cardiac disease', 'heart disease', 'cardiac'],
            'percentage': 45,
            'reason': 'CS is sometimes required if cardiac condition worsens; assisted vaginal preferred otherwise',
            'confidence': 'moderate'
        },
        {
            'field': 'menternal_medical',
            'condition': 'hiv',
            'keywords': ['hiv', 'immunocompromised', 'aids'],
            'percentage': 20,
            'reason': 'Planned CS reduces HIV transmission if viral load high; vaginal possible if undetectable',
            'confidence': 'low'
        },
        {
            'field': 'obstetric_history',
            'condition': 'uterine_rupture',
            'keywords': ['uterine rupture', 'rupture'],
            'percentage': 100,
            'reason': 'Previous rupture mandates scheduled CS before labor',
            'confidence': 'high'
        },
        {
            'field': 'current_pregnancy_menternal',
            'condition': 'ivf_icsi',
            'keywords': ['ivf', 'icsi', 'in vitro', 'assisted reproduction'],
            'percentage': 85,
            'reason': 'Pregnancies achieved via IVF/ICSI show higher rates of elective and emergency Caesarean delivery due to obstetric risks, clinician/patient preference, and precaution for "precious baby" effect',
            'confidence': 'high'
        }
    ]
    
    # CTG-specific rules
    CTG_RULES = {
        'category_ii_suspicious': {
            'percentage': 70,
            'reason': 'Category II often monitored but may require CS if unresolved',
            'confidence': 'moderate'
        },
        'category_iii_pathological': {
            'percentage': 95,
            'reason': 'Category III = urgent CS for suspected hypoxia/acidosis',
            'confidence': 'high'
        }
    }
    
    def __init__(self):
        self.model_dir = Path('ml_models/cs_prediction')
        self.model = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load trained ML model"""
        try:
            model_path = self.model_dir / 'cs_model.pkl'
            metadata_path = self.model_dir / 'model_metadata.json'
            
            if model_path.exists():
                self.model = joblib.load(model_path)
                
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.feature_names = metadata.get('feature_names', [])
        except Exception as e:
            print(f"Warning: Could not load ML model: {e}")
    
    def check_field_condition(self, field_value, keywords, count=None, min_count=None):
        """Check if condition exists in JSON field"""
        if not field_value or field_value == '[]':
            return False
        
        if isinstance(field_value, str):
            try:
                field_value = json.loads(field_value)
            except:
                field_value = [field_value]
        
        if not isinstance(field_value, list):
            return False
        
        field_lower = [str(item).lower() for item in field_value]
        matches = sum(any(kw in item for kw in keywords) for item in field_lower)
        
        if count is not None:
            return matches == count
        elif min_count is not None:
            return matches >= min_count
        else:
            return matches > 0
    
    def check_direct_rules(self, patient):
        """Check direct rules first"""
        matched_rules = []
        
        # Check standard rules
        for rule in self.DIRECT_RULES:
            field_value = getattr(patient, rule['field'], None)
            
            if 'count' in rule:
                if self.check_field_condition(field_value, rule['keywords'], count=rule['count']):
                    matched_rules.append(rule)
            elif 'min_count' in rule:
                if self.check_field_condition(field_value, rule['keywords'], min_count=rule['min_count']):
                    matched_rules.append(rule)
            else:
                if self.check_field_condition(field_value, rule['keywords']):
                    matched_rules.append(rule)
        
        # Check fetal weight rule
        if patient.estimated_fetal_weight_by_gm and patient.estimated_fetal_weight_by_gm >= 4000:
            matched_rules.append({
                'condition': 'macrosomia',
                'percentage': 50,
                'reason': 'Large babies linked to CPD, shoulder dystocia, higher CS rate',
                'confidence': 'moderate'
            })
        
        # Check CTG category
        if patient.ctg_category:
            ctg_key = patient.ctg_category.lower().replace(' ', '_').replace('–', '')
            for key, rule in self.CTG_RULES.items():
                if key in ctg_key:
                    matched_rules.append({
                        'condition': key,
                        **rule
                    })
        
        # Check severe anemia by Hb
        if patient.hb_g_dl and patient.hb_g_dl < 7:
            matched_rules.append({
                'condition': 'severe_anemia_hb',
                'percentage': 25,
                'reason': 'Severe anemia (Hb<7) limits tolerance for labor; CS often chosen if complications exist',
                'confidence': 'low'
            })
        
        return matched_rules
    
    def calculate_combined_probability(self, matched_rules):
        """Calculate combined probability from multiple rules"""
        if not matched_rules:
            return 0, 'low'
        
        # Sort by percentage descending
        matched_rules.sort(key=lambda x: x['percentage'], reverse=True)
        
        # Take highest probability rule
        base_prob = matched_rules[0]['percentage']
        
        # If multiple high-risk factors, increase slightly
        if len(matched_rules) > 1:
            high_risk_count = sum(1 for r in matched_rules if r['percentage'] >= 80)
            if high_risk_count > 1:
                base_prob = min(99, base_prob + 5)
        
        # Determine confidence
        if base_prob >= 85:
            confidence = 'high'
        elif base_prob >= 50:
            confidence = 'moderate'
        else:
            confidence = 'low'
        
        return base_prob, confidence
    
    def prepare_ml_features(self, patient):
        """Prepare features for ML model"""
        if not self.model or not self.feature_names:
            return None, []
        
        features = []
        missing_fields = []
        
        try:
            # Age
            if patient.age:
                features.append(patient.age)
            else:
                features.append(30)  # median
                missing_fields.append('age')
            
            # BMI
            bmi = patient.bmi if patient.bmi else 25
            if not patient.bmi:
                missing_fields.append('bmi')
            features.append(bmi)
            
            # BMI categories
            features.append(1 if 35 <= bmi < 40 else 0)
            features.append(1 if bmi >= 40 else 0)
            
            # Chronic hypertension
            features.append(self.check_field_condition(
                patient.menternal_medical,
                ['chronic hypertension', 'hypertension']
            ))
            
            # Diabetes
            features.append(self.check_field_condition(
                patient.menternal_medical,
                ['diabetes', 'dm', 'gestational diabetes']
            ))
            
            # Grand multipara
            features.append(self.check_field_condition(
                patient.social,
                ['grand multipara', 'multipara']
            ))
            
            # Non-cephalic presentation
            features.append(0 if patient.presentation in ['cephlic', None, ''] else 1)
            
            # Multiple gestation
            features.append(1 if patient.fetus_number in ['twin', 'triplete'] else 0)
            
            # Cervical dilatation
            features.append(patient.cervical_dilatation_at_admission or 0)
            if not patient.cervical_dilatation_at_admission:
                missing_fields.append('cervical_dilatation_at_admission')
            
            # Fetal weight
            efw = patient.estimated_fetal_weight_by_gm or 3000
            if not patient.estimated_fetal_weight_by_gm:
                missing_fields.append('estimated_fetal_weight_by_gm')
            features.append(efw)
            features.append(1 if efw >= 4000 else 0)
            
            # Labor duration
            features.append(patient.labor_duration_hours or 0)
            
            return np.array(features).reshape(1, -1), missing_fields
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return None, missing_fields
    
    def predict_with_ml(self, patient):
        """Predict using ML model"""
        X, missing_fields = self.prepare_ml_features(patient)
        
        if X is None:
            return None, None, missing_fields
        
        try:
            probability = self.model.predict_proba(X)[0][1] * 100
            
            # IMPORTANT: Cap prediction at 6.5% when no direct rules matched
            if probability >= 7:
                probability = 6.5
            
            # Determine confidence based on probability
            if probability >= 70:
                confidence = 'high'
            elif probability >= 40:
                confidence = 'moderate'
            else:
                confidence = 'low'
            
            # Get feature importance for this prediction
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            
            model_details = {
                'model_type': 'Random Forest',
                'top_contributing_features': [f[0] for f in top_features],
                'missing_fields': missing_fields
            }
            
            return probability, confidence, model_details
            
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            return None, None, missing_fields


def predict_patient_by_identifier(patient_identifier):
    """
    Main API function to predict CS probability for a patient
    
    Args:
        patient_identifier: patient_id or file_number
        
    Returns:
        dict: Prediction response
    """
    try:
        # Get patient
        patient = Patient.objects.filter(
            models.Q(patient_id=patient_identifier) | 
            models.Q(file_number=patient_identifier)
        ).first()
        
        if not patient:
            return {
                'success': False,
                'error': f'Patient not found with identifier: {patient_identifier}',
                'patient_id': None,
                'file_number': None,
                'patient_name': None,
                'cs_probability': None,
                'prediction_method': None,
                'reason': None,
                'confidence': None,
                'risk_factors': [],
                'all_reasons': [],
                'model_details': None
            }
        
        # Initialize predictor
        predictor = CSPredictor()
        
        # Check direct rules first
        matched_rules = predictor.check_direct_rules(patient)
        
        if matched_rules:
            # Use direct rules
            probability, confidence = predictor.calculate_combined_probability(matched_rules)
            risk_factors = [rule['condition'] for rule in matched_rules]
            all_reasons = [rule['reason'] for rule in matched_rules]
            reason = '; '.join(all_reasons)
            
            return {
                'success': True,
                'patient_id': patient.patient_id,
                'file_number': patient.file_number,
                'patient_name': patient.name,
                'cs_probability': round(probability, 1),
                'prediction_method': 'direct_rule',
                'reason': reason,
                'confidence': confidence,
                'risk_factors': risk_factors,
                'all_reasons': all_reasons,
                'model_details': None,
                'error': None
            }
        
        # If no direct rules matched, try ML model
        if predictor.model:
            probability, confidence, model_details = predictor.predict_with_ml(patient)
            
            if probability is not None:
                confidence = "low"
                reason = 'No significant risk factors identified; low baseline CS risk based on patient profile'


                return {
                    'success': True,
                    'patient_id': patient.patient_id,
                    'file_number': patient.file_number,
                    'patient_name': patient.name,
                    'cs_probability': round(probability, 1),
                    'prediction_method': 'ml_model',
                    'reason': reason,
                    'confidence': confidence,
                    'risk_factors': [],
                    'all_reasons': [reason],
                    'model_details': model_details,
                    'error': None
                }
        
        # If no model available and no direct rules, return low baseline
        return {
            'success': True,
            'patient_id': patient.patient_id,
            'file_number': patient.file_number,
            'patient_name': patient.name,
            'cs_probability': 6.0,
            'prediction_method': 'baseline',
            'reason': 'No specific risk factors identified; returning baseline population CS rate',
            'confidence': 'low',
            'risk_factors': [],
            'all_reasons': ['No specific risk factors identified; returning baseline population CS rate'],
            'model_details': None,
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Prediction error: {str(e)}',
            'patient_id': patient_identifier,
            'file_number': None,
            'patient_name': None,
            'cs_probability': None,
            'prediction_method': None,
            'reason': None,
            'confidence': None,
            'risk_factors': [],
            'all_reasons': [],
            'model_details': None
        }


# Django REST API View (optional - for easy integration)
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny

@api_view(['GET', 'POST'])
@permission_classes([AllowAny])
def predict_cs_api_view(request):
    """
    API endpoint for CS prediction
    GET: /api/predict-cs/?patient_id=123
    POST: /api/predict-cs/ with body {"patient_id": "123"}
    """
    if request.method == 'GET':
        patient_identifier = request.GET.get('patient_id') or request.GET.get('file_number')
    else:
        patient_identifier = request.data.get('patient_id') or request.data.get('file_number')
    
    if not patient_identifier:
        return Response(
            {'error': 'patient_id or file_number is required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    result = predict_patient_by_identifier(patient_identifier)
    
    if result['success']:
        return Response(result, status=status.HTTP_200_OK)
    else:
        return Response(result, status=status.HTTP_404_NOT_FOUND)
