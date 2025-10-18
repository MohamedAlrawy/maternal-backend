"""
PPH Prediction Service
Location: yourapp/services/pph_predictor.py

Usage in views:
from yourapp.services.pph_predictor import PPHPredictor
predictor = PPHPredictor()
result = predictor.predict_patient_by_identifier(patient_id)
"""

import json
import pickle
import numpy as np
from pathlib import Path
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from patients.models import Patient
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny


class PPHPredictor:
    """PPH Prediction Service"""
    
    # Direct risk factors with percentage and reason
    DIRECT_RISK_FACTORS = {
        'prev_pph': {
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
        'preeclampsia': {
            'percentage': 7,
            'reason': 'Preeclampsia associated with endothelial dysfunction & coagulopathy'
        },
        'prolonged_labor_history': {
            'percentage': 7,
            'reason': 'Prolonged labor leads to uterine exhaustion and atony'
        },
    }
    
    def __init__(self):
        """Initialize predictor with trained model"""
        self.models_dir = Path(settings.BASE_DIR) / 'ml_models'
        self.model = None
        self.scaler = None
        self.metrics = None
        self.feature_names = None
        self._load_model()
    
    def _load_model(self):
        """Load trained model and scaler"""
        try:
            model_path = self.models_dir / 'pph_model.pkl'
            scaler_path = self.models_dir / 'pph_scaler.pkl'
            metrics_path = self.models_dir / 'pph_metrics.json'
            
            if not model_path.exists():
                raise FileNotFoundError(f'Model file not found: {model_path}')
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)
                    self.feature_names = self.metrics.get('feature_names', [])
        except Exception as e:
            raise RuntimeError(f'Failed to load model: {str(e)}')
    
    def predict_patient_by_identifier(self, patient_id):
        """
        Predict PPH risk for a patient
        
        Args:
            patient_id: Patient ID string
            
        Returns:
            dict: {
                'success': bool,
                'risk_percentage': float (0-100),
                'risk_level': str (Low/Moderate/High),
                'reasons': list,
                'direct_factors': list,
                'model_prediction': dict (if model used),
                'error': str (if failed)
            }
        """
        result = {
            'success': False,
            'risk_percentage': 0,
            'risk_level': 'Unknown',
            'reasons': [],
            'direct_factors': [],
            'model_prediction': None,
            'error': None
        }
        
        try:
            # Fetch patient
            try:
                patient = Patient.objects.get(file_number=patient_id)
            except ObjectDoesNotExist:
                result['error'] = f'Patient with ID {patient_id} not found'
                return result
            
            # Step 1: Check direct risk factors
            direct_risk = self._check_direct_factors(patient)
            result['direct_factors'] = direct_risk['factors']
            
            if direct_risk['total_percentage'] > 0:
                result['risk_percentage'] = min(direct_risk['total_percentage'], 95)
                result['reasons'] = direct_risk['reasons']
                result['risk_level'] = self._get_risk_level(result['risk_percentage'])
                result['success'] = True
                return result
            
            # Step 2: If no direct factors, use ML model
            if not self.model:
                result['error'] = 'ML model not available and no direct factors found'
                return result
            
            features = self._extract_features(patient)
            
            if not features or all(v == 0 for v in features.values()):
                result['error'] = 'Insufficient data for prediction'
                return result
            
            # Prepare feature vector
            feature_vector = np.array([
                features.get(name, 0) for name in self.feature_names
            ]).reshape(1, -1)
            
            # Scale and predict
            feature_vector_scaled = self.scaler.transform(feature_vector)
            risk_proba = self.model.predict_proba(feature_vector_scaled)[0][1]
            
            result['risk_percentage'] = round(risk_proba * 100, 1)
            result['risk_level'] = self._get_risk_level(result['risk_percentage'])
            result['reasons'] = [
                'Prediction based on machine learning model trained on patient cohort'
            ]
            result['model_prediction'] = {
                'probability': round(risk_proba, 4),
                'auc': self.metrics.get('auc', 'N/A') if self.metrics else 'N/A',
                'sensitivity': self.metrics.get('sensitivity', 'N/A') if self.metrics else 'N/A',
                'specificity': self.metrics.get('specificity', 'N/A') if self.metrics else 'N/A',
            }
            result['success'] = True
            return result
            
        except Exception as e:
            result['error'] = f'Prediction failed: {str(e)}'
            return result
    
    def _check_direct_factors(self, patient):
        """Check for direct risk factors"""
        factors = []
        reasons = []
        total_percentage = 0
        
        # Check previous PPH
        if self._check_in_json(patient.obstetric_history, ['History of postpartum hemorrhage', 'PPH']):
            factors.append('Previous PPH')
            reasons.append(self.DIRECT_RISK_FACTORS['prev_pph']['reason'])
            total_percentage += self.DIRECT_RISK_FACTORS['prev_pph']['percentage']
        
        # Check multiple gestation in current pregnancy
        if self._check_in_json(patient.current_pregnancy_menternal, ['Multiple gestation']):
            factors.append('Multiple gestation')
            reasons.append(self.DIRECT_RISK_FACTORS['multiple_gestation_current']['reason'])
            total_percentage += self.DIRECT_RISK_FACTORS['multiple_gestation_current']['percentage']
        
        # Check grand multipara
        if patient.parity and patient.parity >= 5:
            factors.append('Grand multipara (≥5 births)')
            reasons.append(self.DIRECT_RISK_FACTORS['grand_multipara']['reason'])
            total_percentage += self.DIRECT_RISK_FACTORS['grand_multipara']['percentage']
        
        # Check large baby
        if patient.estimated_fetal_weight_by_gm and patient.estimated_fetal_weight_by_gm >= 4000:
            factors.append('Large baby (≥4000g)')
            reasons.append(self.DIRECT_RISK_FACTORS['large_baby']['reason'])
            total_percentage += self.DIRECT_RISK_FACTORS['large_baby']['percentage']
        
        # Check polyhydramnios
        if patient.liquor == 'polihydraminos':
            factors.append('Polyhydramnios')
            reasons.append(self.DIRECT_RISK_FACTORS['polyhydramnios']['reason'])
            total_percentage += self.DIRECT_RISK_FACTORS['polyhydramnios']['percentage']
        
        # Check placental abruption (current)
        if self._check_in_json(patient.current_pregnancy_menternal, ['Placental abruption']):
            factors.append('Placental abruption (current)')
            reasons.append(self.DIRECT_RISK_FACTORS['placenta_abruption_current']['reason'])
            total_percentage += self.DIRECT_RISK_FACTORS['placenta_abruption_current']['percentage']
        
        # Check placental abruption (history)
        if self._check_in_json(patient.obstetric_history, ['Placental abruption']):
            factors.append('Placental abruption (history)')
            reasons.append(self.DIRECT_RISK_FACTORS['placenta_abruption_history']['reason'])
            total_percentage += self.DIRECT_RISK_FACTORS['placenta_abruption_history']['percentage']
        
        # Check placenta previa (current)
        if self._check_in_json(patient.current_pregnancy_menternal, ['Placenta previa']):
            factors.append('Placenta previa (current)')
            reasons.append(self.DIRECT_RISK_FACTORS['placenta_previa_current']['reason'])
            total_percentage += self.DIRECT_RISK_FACTORS['placenta_previa_current']['percentage']
        
        # Check placenta previa (history)
        if self._check_in_json(patient.obstetric_history, ['Placenta previa']):
            factors.append('Placenta previa (history)')
            reasons.append(self.DIRECT_RISK_FACTORS['placenta_previa_history']['reason'])
            total_percentage += self.DIRECT_RISK_FACTORS['placenta_previa_history']['percentage']
        
        # Check preeclampsia
        if self._check_in_json(patient.current_pregnancy_menternal, ['Pre-eclampsia', 'Preeclampsia']):
            factors.append('Pre-eclampsia')
            reasons.append(self.DIRECT_RISK_FACTORS['preeclampsia']['reason'])
            total_percentage += self.DIRECT_RISK_FACTORS['preeclampsia']['percentage']
        
        # Check prolonged labor (history)
        if self._check_in_json(patient.obstetric_history, ['Obstructed/prolonged labor', 'Prolonged labor']):
            factors.append('Prolonged labor history')
            reasons.append(self.DIRECT_RISK_FACTORS['prolonged_labor_history']['reason'])
            total_percentage += self.DIRECT_RISK_FACTORS['prolonged_labor_history']['percentage']
        
        return {
            'factors': factors,
            'reasons': reasons,
            'total_percentage': min(total_percentage, 95)
        }
    
    def _extract_features(self, patient):
        """Extract features from patient for ML model"""
        try:
            features = {}
            
            features['age'] = patient.age or 0
            features['parity'] = patient.parity or 0
            features['bmi'] = float(patient.bmi) if patient.bmi else 0
            
            features['pulse'] = patient.pulse or 0
            try:
                bp_systolic = int(patient.bp.split('/')[0]) if patient.bp else 0
                features['bp_systolic'] = bp_systolic
            except:
                features['bp_systolic'] = 0
            
            features['hb'] = patient.hb_g_dl or 0
            features['platelets'] = patient.platelets_x10e9l or 0
            
            features['cervical_dilatation_at_admission'] = patient.cervical_dilatation_at_admission or 0
            features['labor_duration_hours'] = patient.labor_duration_hours or 0
            features['estimated_fetal_weight'] = patient.estimated_fetal_weight_by_gm or 0
            
            features['chronic_hypertension'] = self._check_in_json(patient.menternal_medical, ['Chronic hypertension'])
            features['diabetes'] = self._check_in_json(patient.menternal_medical, ['Diabetes', 'Gestational diabetes'])
            features['blood_transfusion_history'] = self._check_in_json(patient.menternal_medical, ['History of blood transfusion'])
            features['severe_anemia'] = self._check_in_json(patient.current_pregnancy_menternal, ['Severe anemia'])
            
            features['prev_cs_single'] = self._check_in_json(patient.obstetric_history, ['Previous c-section'])
            features['prev_cs_multiple'] = self._check_in_json(patient.obstetric_history, ['Multiple c-sections'])
            features['prev_pph'] = self._check_in_json(patient.obstetric_history, ['History of postpartum hemorrhage'])
            features['prolonged_labor_history'] = self._check_in_json(patient.obstetric_history, ['Obstructed/prolonged labor'])
            
            features['placenta_abruption_current'] = self._check_in_json(patient.current_pregnancy_menternal, ['Placental abruption'])
            features['placenta_previa_current'] = self._check_in_json(patient.current_pregnancy_menternal, ['Placenta previa'])
            features['preeclampsia'] = self._check_in_json(patient.current_pregnancy_menternal, ['Pre-eclampsia'])
            features['multiple_gestation'] = self._check_in_json(patient.current_pregnancy_menternal, ['Multiple gestation'])
            
            features['grand_multipara'] = 1 if patient.parity >= 5 else 0
            features['bmi_obese_35_39'] = 1 if 35 <= float(patient.bmi or 0) < 39.5 else 0
            features['bmi_obese_40_plus'] = 1 if float(patient.bmi or 0) >= 40 else 0
            features['fetus_number_twin'] = 1 if patient.fetus_number == 'twin' else 0
            features['fetus_number_triplet'] = 1 if patient.fetus_number == 'triplete' else 0
            
            features['instrumental_delivery'] = int(patient.instrumental_delivery)
            features['cs_emergency'] = 1 if patient.type_of_cs == 'emergency' else 0
            features['mode_cs'] = 1 if patient.mode_of_delivery == 'cs' else 0
            features['polyhydramnios'] = 1 if patient.liquor == 'polihydraminos' else 0
            
            features['ctg_category_ii'] = 1 if patient.ctg_category == 'category_ii_suspicious' else 0
            features['ctg_category_iii'] = 1 if patient.ctg_category == 'category_iii_pathological' else 0
            
            features['placenta_lower'] = 1 if patient.placenta_location == 'lower' else 0
            features['placenta_covering_cervix'] = 1 if patient.placenta_location == 'covering_cervix' else 0
            
            return features
        except Exception as e:
            return None
    
    def _check_in_json(self, json_field, keywords):
        """Check if any keyword exists in JSON field"""
        if not json_field:
            return 0
        try:
            if isinstance(json_field, str):
                data = json.loads(json_field)
            else:
                data = json_field
            
            if isinstance(data, list):
                text = ' '.join(data).lower()
            else:
                text = str(data).lower()
            
            for keyword in keywords:
                if keyword.lower() in text:
                    return 1
            return 0
        except:
            return 0
    
    def _get_risk_level(self, percentage):
        """Determine risk level from percentage"""
        if percentage < 20:
            return 'Low'
        elif percentage < 50:
            return 'Moderate'
        else:
            return 'High'


# API View Example

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
# from .services.pph_predictor import PPHPredictor
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny


@require_http_methods(["GET"])
@permission_classes([AllowAny])
def predict_pph(request):
    print("predict_pph")
    try:
        patient_id = request.GET.get('patient_id')
        
        if not patient_id:
            return JsonResponse({
                'success': False,
                'error': 'patient_id is required'
            }, status=400)
        
        predictor = PPHPredictor()
        result = predictor.predict_patient_by_identifier(patient_id)
        
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
