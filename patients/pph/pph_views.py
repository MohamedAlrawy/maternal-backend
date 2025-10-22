"""
File: patients/views.py or patients/pph/pph_views.py - PPH Prediction API and Service
"""
import os
import json
import pickle
import numpy as np
from django.conf import settings
from django.db.models import Q
from rest_framework.decorators import api_view, permission_classes  
from rest_framework.response import Response
from rest_framework.permissions import AllowAny


class PPHPredictionService:
    """Service for PPH prediction"""
    
    # Direct rule-based risk factors
    DIRECT_FACTORS = {
        'pph_history': {
            'percentage': 20,
            'reason': 'Previous PPH strongly predicts recurrence'
        },
        'multiple_gestation_current': {
            'percentage': 15,
            'reason': 'Overdistended uterus after twins/triplets increases atony risk'
        },
        'grand_multipara': {
            'percentage': 15,
            'reason': 'Uterine overdistension reduces contractility leading to atony'
        },
        'large_baby': {
            'percentage': 10,
            'reason': 'Large baby overstretches uterus leading to atony'
        },
        'polyhydramnios': {
            'percentage': 9,
            'reason': 'Excess fluid overstretches uterus causing poor contraction'
        },
        'placenta_abruption_current': {
            'percentage': 12,
            'reason': 'Associated with coagulopathy and severe hemorrhage'
        },
        'placenta_abruption_history': {
            'percentage': 12,
            'reason': 'Associated with coagulopathy and severe hemorrhage'
        },
        'placenta_previa_current': {
            'percentage': 15,
            'reason': 'Placenta previa increases postpartum bleeding risk'
        },
        'placenta_previa_history': {
            'percentage': 15,
            'reason': 'Placenta previa increases postpartum bleeding risk'
        },
        'severe_anemia': {
            'percentage': 10,
            'reason': 'Severe anemia reduces oxygen-carrying capacity'
        },
        'preeclampsia': {
            'percentage': 7,
            'reason': 'Preeclampsia associated with endothelial dysfunction & coagulopathy'
        },
        'multiple_gestation_adhesions': {
            'percentage': 7,
            'reason': 'Adhesions and abnormal placentation increase bleeding risk'
        },
        'prolonged_labor': {
            'percentage': 7,
            'reason': 'Prolonged labor leads to uterine exhaustion and atony'
        },
        'uterine_rupture_history': {
            'percentage': 50,
            'reason': 'Uterine rupture history significantly increases PPH risk and recurrence'
        },
        'polyhydraminos_current': {
            'percentage': 10,
            'reason': 'Excess amniotic fluid in current pregnancy increases uterine overdistension risk'
        }
    }

    def __init__(self):
        self.model_dir = os.path.join(settings.BASE_DIR, 'ml_models', 'pph')
        self.models_loaded = False
        self.rf_model = None
        self.scaler = None
        self.imputer = None
        self.report = None
        self._load_models()

    def _load_models(self):
        """Load trained models from disk"""
        try:
            if not os.path.exists(self.model_dir):
                return
            
            model_path = os.path.join(self.model_dir, 'rf_model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.rf_model = pickle.load(f)
                with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
                    self.scaler = pickle.load(f)
                with open(os.path.join(self.model_dir, 'imputer.pkl'), 'rb') as f:
                    self.imputer = pickle.load(f)
                with open(os.path.join(self.model_dir, 'report.json'), 'r') as f:
                    self.report = json.load(f)
                self.models_loaded = True
        except Exception as e:
            print(f"Error loading models: {str(e)}")

    def predict_patient_by_identifier(self, patient_identifier):
        """
        Main prediction function - takes patient ID or file number
        Returns prediction with reasoning
        """
        try:
            from patients.models import Patient
            
            # Fetch patient
            patient = Patient.objects.filter(
                Q(patient_id=patient_identifier) | 
                Q(file_number=patient_identifier)
            ).first()
            
            if not patient:
                return self._error_response(
                    f'Patient {patient_identifier} not found',
                    patient_id=patient_identifier
                )
            
            patient_dict = self._patient_to_dict(patient)
            
            # Try direct rule-based prediction first
            direct_result = self._predict_direct_rules(patient_dict)
            if direct_result['success']:
                direct_result.update({
                    'patient_id': patient.patient_id,
                    'file_number': patient.file_number,
                    'patient_name': patient.name,
                })
                return direct_result
            
            # Fall back to ML model
            if not self.models_loaded:
                return self._error_response(
                    'ML model not available. Train model using: python manage.py train_pph_model',
                    patient_id=patient.patient_id
                )
            
            ml_result = self._predict_ml_model(patient_dict)
            ml_result.update({
                'patient_id': patient.patient_id,
                'file_number': patient.file_number,
                'patient_name': patient.name,
            })
            return ml_result
            
        except Exception as e:
            return self._error_response(f'Prediction error: {str(e)}')

    def _predict_direct_rules(self, patient):
        """Apply direct rule-based prediction"""
        detected_factors = []
        total_percentage = 0
        reasons = []
        groups = {
            'severe': [],
            'moderate': [],
            'mild': []
        }
        
        # Check PPH history
        if self._check_in_list(patient.get('obstetric_history', []), ['History of Postpartum hemorrhage']):
            detected_factors.append('pph_history')
            total_percentage = self._combine_probability(total_percentage, 20)
            reasons.append(self.DIRECT_FACTORS['pph_history']['reason'])
            groups['severe'].append('pph_history')
        
        # Check multiple gestation
        if patient.get('fetus_number') in ['twin', 'triplete']:
            detected_factors.append('multiple_gestation_current')
            total_percentage = self._combine_probability(total_percentage, 15)
            reasons.append(self.DIRECT_FACTORS['multiple_gestation_current']['reason'])
            groups['severe'].append('multiple_gestation_current')
        
        # Check grand multipara
        if (patient.get('parity') or 0) >= 5:
            detected_factors.append('grand_multipara')
            total_percentage = self._combine_probability(total_percentage, 15)
            reasons.append(self.DIRECT_FACTORS['grand_multipara']['reason'])
            groups['severe'].append('grand_multipara')
        
        # Check placenta abruption current
        if self._check_in_list(patient.get('current_pregnancy_menternal', []), ['Placenta abruption']):
            detected_factors.append('placenta_abruption_current')
            total_percentage = self._combine_probability(total_percentage, 12)
            reasons.append(self.DIRECT_FACTORS['placenta_abruption_current']['reason'])
            groups['severe'].append('placenta_abruption_current')
        
        # Check placenta abruption history
        if self._check_in_list(patient.get('obstetric_history', []), ['Placenta abruption']):
            detected_factors.append('placenta_abruption_history')
            total_percentage = self._combine_probability(total_percentage, 12)
            reasons.append(self.DIRECT_FACTORS['placenta_abruption_history']['reason'])
            groups['severe'].append('placenta_abruption_history')
        
        # Check placenta previa current
        if self._check_in_list(patient.get('current_pregnancy_menternal', []), ['Placenta previa']) or \
           patient.get('placenta_location') == 'covering_cervix':
            detected_factors.append('placenta_previa_current')
            total_percentage = self._combine_probability(total_percentage, 15)
            reasons.append(self.DIRECT_FACTORS['placenta_previa_current']['reason'])
            groups['severe'].append('placenta_previa_current')
        
        # Check placenta previa history
        if self._check_in_list(patient.get('obstetric_history', []), ['Placenta previa']):
            detected_factors.append('placenta_previa_history')
            total_percentage = self._combine_probability(total_percentage, 15)
            reasons.append(self.DIRECT_FACTORS['placenta_previa_history']['reason'])
            groups['severe'].append('placenta_previa_history')
        
        # Check large baby
        if (patient.get('estimated_fetal_weight_by_gm') or 0) >= 4000:
            detected_factors.append('large_baby')
            total_percentage = self._combine_probability(total_percentage, 10)
            reasons.append(self.DIRECT_FACTORS['large_baby']['reason'])
            groups['moderate'].append('large_baby')
        
        # Check polyhydramnios
        if patient.get('liquor') in ['polihydraminos', 'Polyhydramnios']:
            detected_factors.append('polyhydramnios')
            total_percentage = self._combine_probability(total_percentage, 9)
            reasons.append(self.DIRECT_FACTORS['polyhydramnios']['reason'])
            groups['moderate'].append('polyhydramnios')
        
        # Check severe anemia
        if (patient.get('hb_g_dl') or 100) < 7:
            detected_factors.append('severe_anemia')
            total_percentage = self._combine_probability(total_percentage, 10)
            reasons.append(self.DIRECT_FACTORS['severe_anemia']['reason'])
            groups['moderate'].append('severe_anemia')
        
        # Check preeclampsia
        if self._check_in_list(patient.get('current_pregnancy_menternal', []), ['Pre-eclampsia']):
            detected_factors.append('preeclampsia')
            total_percentage = self._combine_probability(total_percentage, 7)
            reasons.append(self.DIRECT_FACTORS['preeclampsia']['reason'])
            groups['mild'].append('preeclampsia')
        
        # Check prolonged labor
        if (patient.get('labor_duration_hours') or 0) > 12 or \
           self._check_in_list(patient.get('obstetric_history', []), ['Obstructed/prolonged labor']):
            detected_factors.append('prolonged_labor')
            total_percentage = self._combine_probability(total_percentage, 7)
            reasons.append(self.DIRECT_FACTORS['prolonged_labor']['reason'])
            groups['mild'].append('prolonged_labor')
        
        if self._check_in_list(patient.get('obstetric_history', []), ['Uterine rupture']):
            detected_factors.append('uterine_rupture_history')
            total_percentage = self._combine_probability(total_percentage, 50)
            reasons.append(self.DIRECT_FACTORS['uterine_rupture_history']['reason'])
            groups['severe'].append('uterine_rupture_history')
        
        # Check polyhydramnios in current pregnancy
        if self._check_in_list(patient.get('current_pregnancy_menternal', []), ['Polyhydramnios', 'polihydraminos']):
            detected_factors.append('polyhydraminos_current')
            total_percentage = self._combine_probability(total_percentage, 10)
            reasons.append(self.DIRECT_FACTORS['polyhydraminos_current']['reason'])
            groups['moderate'].append('polyhydraminos_current')
        
        if not detected_factors:
            return {'success': False}
        
        confidence = self._get_confidence_level(len(detected_factors), total_percentage)
        
        return {
            'success': True,
            'pph_probability': round(total_percentage, 1),
            'prediction_method': 'direct_rule',
            'reason': '; '.join(reasons),
            'confidence': confidence,
            'risk_factors': self._limit_list(detected_factors),
            'all_reasons': reasons,
            'risk_groups': {
                'severe_risk': self._limit_list(groups['severe']),
                'moderate_risk': self._limit_list(groups['moderate']),
                'mild_risk': self._limit_list(groups['mild'])
            },
            'model_details': None,
            'error': None
        }

    def _predict_ml_model(self, patient):
        """Apply ML model prediction"""
        try:
            feature_vector = self._prepare_features(patient)
            
            if feature_vector is None:
                return {'success': False, 'error': 'Insufficient data for ML prediction'}
            
            feature_vector = self.imputer.transform([feature_vector])[0]
            feature_vector_scaled = self.scaler.transform([feature_vector])[0]
            
            # Predict
            probability = self.rf_model.predict_proba([feature_vector_scaled])[0][1]
            percentage = round(probability * 100, 1)
            
            # REDUCE PREDICTION: Apply calibration factor
            # This reduces overconfident predictions
            percentage = self._calibrate_prediction(percentage)
            
            confidence = self._get_confidence_level_ml(percentage)
            
            # Generate reason based on probability level
            if percentage >= 8:
                reason = 'Elevated PPH risk - ML model indicates potential concerns'
            elif percentage >= 5:
                reason = 'Low-moderate PPH risk detected by ML model'
            else:
                reason = 'Low PPH risk based on available clinical factors'
            
            return {
                'success': True,
                'pph_probability': percentage,
                'prediction_method': 'machine_learning',
                'reason': reason,
                'confidence': confidence,
                'model_details': {
                    'auc': self.report.get('auc') if self.report else None,
                    'sensitivity': self.report.get('sensitivity') if self.report else None,
                    'specificity': self.report.get('specificity') if self.report else None,
                },
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'ML prediction error: {str(e)}'
            }

    def _calibrate_prediction(self, percentage):
        """
        Heavily reduce ML model predictions - very conservative approach
        Only high predictions are allowed, most are capped at 8%
        """
        
        # AGGRESSIVE REDUCTION for ML predictions without direct rules
        REDUCTION_FACTOR = 0.15  # Reduces by 85% - very aggressive
        reduced = percentage * REDUCTION_FACTOR
        
        # Apply floor threshold (min 2%)
        MIN_THRESHOLD = 2.0
        reduced = max(reduced, MIN_THRESHOLD)
        
        # CAP AT 8% - MAXIMUM prediction without direct rules
        # This means ML-only predictions will almost never exceed 8%
        MAX_THRESHOLD = 8.0
        reduced = min(reduced, MAX_THRESHOLD)
        
        return round(reduced, 1)

    def _prepare_features(self, patient):
        """Prepare feature vector for ML model"""
        try:
            features = []
            
            # GROUP 1: Demographics & Lab
            features.append(patient.get('age') or 0)
            features.append(patient.get('bmi') or 0)
            features.append(1 if self._check_in_list(
                patient.get('menternal_medical', []),
                ['Chronic hypertension', 'History of blood transfusion']
            ) else 0)
            features.append(patient.get('hb_g_dl') or 0)
            features.append(patient.get('platelets_x10e9l') or 0)
            
            # GROUP 2: Pregnancy factors
            features.append(patient.get('parity') or 0)
            features.append(1 if self._check_in_list(
                patient.get('obstetric_history', []),
                ['Multiple c-sections (2)', 'Previous c-section (1)', 'Multiple c-sections (>3)']
            ) else 0)
            features.append(1 if patient.get('liquor') in ['polihydraminos', 'Polyhydramnios'] else 0)
            features.append(1 if patient.get('fetus_number') in ['twin', 'triplete'] else 0)
            features.append(self._encode_placenta_location(patient.get('placenta_location')))
            features.append(patient.get('estimated_fetal_weight_by_gm') or 0)
            
            # GROUP 3: Labor factors
            features.append(self._encode_labor_type(patient.get('type_of_labor')))
            features.append(self._encode_cs_type(patient.get('type_of_cs')))
            features.append(patient.get('labor_duration_hours') or 0)
            
            return features
        except:
            return None

    def _patient_to_dict(self, patient):
        """Convert Patient model to dictionary"""
        return {
            'name': patient.name,
            'patient_id': patient.patient_id,
            'file_number': patient.file_number,
            'age': patient.age,
            'bmi': float(patient.bmi) if patient.bmi else 0,
            'hb_g_dl': float(patient.hb_g_dl) if patient.hb_g_dl else 0,
            'platelets_x10e9l': patient.platelets_x10e9l,
            'parity': patient.parity,
            'fetus_number': patient.fetus_number,
            'liquor': patient.liquor,
            'placenta_location': patient.placenta_location,
            'estimated_fetal_weight_by_gm': float(patient.estimated_fetal_weight_by_gm) if patient.estimated_fetal_weight_by_gm else 0,
            'type_of_labor': patient.type_of_labor,
            'type_of_cs': patient.type_of_cs,
            'labor_duration_hours': float(patient.labor_duration_hours) if patient.labor_duration_hours else 0,
            'obstetric_history': patient.obstetric_history or [],
            'menternal_medical': patient.menternal_medical or [],
            'current_pregnancy_menternal': patient.current_pregnancy_menternal or [],
            'blood_loss': patient.blood_loss,
        }

    def _check_in_list(self, json_field, keywords):
        """Check if keyword exists in list/JSON field"""
        if not json_field:
            return False
        try:
            text = str(json_field).lower()
            return any(kw.lower() in text for kw in keywords)
        except:
            return False

    def _combine_probability(self, p1, p2):
        """Combine two probabilities using formula: P_Total = 1 - (1 - P1) × (1 - P2)"""
        p1_decimal = p1 / 100
        p2_decimal = p2 / 100
        combined = 1 - (1 - p1_decimal) * (1 - p2_decimal)
        return combined * 100

    def _encode_placenta_location(self, location):
        """Encode placenta location"""
        encoding = {'upper': 0, 'lower': 1, 'covering_cervix': 2}
        return encoding.get(location, -1)

    def _encode_labor_type(self, labor_type):
        """Encode labor type"""
        encoding = {'spontenous_labor': 0, 'iol': 1, 'pre_labour_cesarean': 2, 'no_labour_pain': 3}
        return encoding.get(labor_type, -1)

    def _encode_cs_type(self, cs_type):
        """Encode CS type"""
        encoding = {'emergency': 1, 'elective': 0}
        return encoding.get(cs_type, -1)

    def _get_confidence_level(self, factor_count, percentage):
        """Get confidence level for direct rules"""
        if factor_count >= 3 and percentage >= 30:
            return 'high'
        elif factor_count >= 2 or percentage >= 20:
            return 'moderate'
        else:
            return 'low'

    def _get_confidence_level_ml(self, percentage):
        """Get confidence level for ML prediction (on reduced scale 0-8%)"""
        if percentage >= 8:
            return 'high'
        elif percentage >= 4:
            return 'moderate'
        else:
            return 'low'

    def _limit_list(self, items):
        """Limit list to maximum 8 items"""
        return items[:8]

    def _error_response(self, error_msg, patient_id=None):
        """Generate error response"""
        return {
            'success': False,
            'patient_id': patient_id,
            'file_number': None,
            'patient_name': None,
            'pph_probability': None,
            'prediction_method': None,
            'reason': None,
            'confidence': None,
            'risk_factors': [],
            'all_reasons': [],
            'model_details': None,
            'error': error_msg
        }


# Initialize service singleton
pph_service = PPHPredictionService()


@api_view(['GET', 'POST'])
@permission_classes([AllowAny])
def predict_pph(request):
    """
    API endpoint for PPH prediction
    GET: /api/predict-pph/?patient_id=161616
    POST: /api/predict-pph/ with body {"patient_id": "161616"}
    """
    try:
        patient_id = None
        
        if request.method == 'GET':
            patient_id = request.query_params.get('patient_id') or request.query_params.get('file_number')
        else:
            patient_id = request.data.get('patient_id') or request.data.get('file_number')
        
        if not patient_id:
            return Response({
                'success': False,
                'error': 'patient_id or file_number required'
            }, status=400)
        
        result = pph_service.predict_patient_by_identifier(patient_id)
        
        # Add patient info if successful
        if result.get('success'):
            # Keep essential fields only for ML predictions
            if result.get('prediction_method') == 'machine_learning':
                result_response = {
                    'success': result['success'],
                    'patient_id': result['patient_id'],
                    'file_number': result['file_number'],
                    'patient_name': result['patient_name'],
                    'pph_probability': result['pph_probability'],
                    'prediction_method': result['prediction_method'],
                    'reason': result['reason'],
                    'confidence': result['confidence'],
                    'model_details': result['model_details'],
                    'error': result['error']
                }
            else:
                # Keep all fields for direct rule predictions
                result_response = {
                    'success': result['success'],
                    'patient_id': result['patient_id'],
                    'file_number': result['file_number'],
                    'patient_name': result['patient_name'],
                    'pph_probability': result['pph_probability'],
                    'prediction_method': result['prediction_method'],
                    'reason': result['reason'],
                    'confidence': result['confidence'],
                    'risk_factors': result['risk_factors'],
                    'risk_groups': result['risk_groups'],
                    'all_reasons': result['all_reasons'],
                    'model_details': result['model_details'],
                    'error': result['error']
                }
        else:
            result_response = result
        
        status_code = 200 if result_response.get('success') else 400
        return Response(result_response, status=status_code)
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=500)


@api_view(['GET'])
@permission_classes([AllowAny])
def get_model_report(request):
    """Get training report and metrics"""
    try:
        if not pph_service.report:
            return Response({
                'success': False,
                'error': 'Model not trained. Run: python manage.py train_pph_model'
            }, status=400)
        
        report = pph_service.report
        significant_features = report.get('significant_features', [])
        
        response_data = {
            'success': True,
            'task': 'Prediction of Postpartum Hemorrhage (PPH)',
            'model_metrics': {
                'auc': report.get('auc', 'N/A'),
                'sensitivity': report.get('sensitivity', 'N/A'),
                'specificity': report.get('specificity', 'N/A'),
                'accuracy': report.get('accuracy', 'N/A')
            },
            'training_data': {
                'training_samples': report.get('training_samples', 0),
                'test_samples': report.get('test_samples', 0),
                'total_samples': report.get('training_samples', 0) + report.get('test_samples', 0)
            },
            'significant_risk_factors': [
                {
                    'feature': f.get('feature'),
                    'importance': round(f.get('importance', 0), 4)
                } for f in significant_features[:10]
            ],
            'model_type': 'Random Forest Classifier',
            'timestamp': report.get('timestamp'),
            'roc_curve_available': os.path.exists(os.path.join(pph_service.model_dir, 'roc_curve.png'))
        }
        
        return Response(response_data, status=200)
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=500)