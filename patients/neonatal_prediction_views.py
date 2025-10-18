# File: your_app/services/neonatal_predictor.py

import os
import json
import pickle
import numpy as np
from decimal import Decimal
from django.apps import apps


class NeonatalPredictionService:
    """Service for predicting neonatal complications"""
    
    # Direct rule-based predictions
    DIRECT_RULES = {
        'category_iii_pathological': {
            'probability': 60,
            'impact': 'NICU/HIE/death',
            'reason': 'Immature lungs/organs increase need for NICU and complications'
        },
        'category_ii_suspicious': {
            'probability': 80,
            'impact': 'HIE/NICU',
            'reason': 'Suggests hypoxia/acidosis → urgent delivery, high neonatal risk'
        },
        'placenta_abruption': {
            'probability': 45,
            'impact': 'HIE/death',
            'reason': 'Acute placental separation → fetal hypoxia'
        },
        'placenta_previa': {
            'probability': 25,
            'impact': 'NICU',
            'reason': 'Antepartum bleeding/preterm delivery risk'
        },
        'multiple_gestation': {
            'probability': 30,
            'impact': 'NICU',
            'reason': 'Prematurity and low birth weight more common'
        },
        'non_cephalic': {
            'probability': 15,
            'impact': 'NICU',
            'reason': 'Higher risk of operative delivery and birth trauma/hypoxia'
        },
        'prolonged_rupture': {
            'probability': 15,
            'impact': 'Sepsis/NICU',
            'reason': 'Infection risk (chorioamnionitis/early-onset sepsis)'
        },
        'prom': {
            'probability': 8,
            'impact': 'Infection',
            'reason': 'Increased ascending infection risk vs intact'
        },
        'preeclampsia': {
            'probability': 25,
            'impact': 'NICU/SGA',
            'reason': 'Uteroplacental insufficiency, indicated preterm birth'
        },
        'diabetes': {
            'probability': 15,
            'impact': 'NICU',
            'reason': 'Macrosomia, hypoglycemia, respiratory distress'
        },
        'severe_anemia': {
            'probability': 10,
            'impact': 'NICU',
            'reason': 'Fetal hypoxia/low reserve'
        },
        'polyhydramios': {
            'probability': 9,
            'impact': 'NICU',
            'reason': 'Associated with anomalies, cord prolapse, malpresentation'
        },
        'macrosomia': {
            'probability': 15,
            'impact': 'Birth injury/NICU',
            'reason': 'Difficult labor, shoulder dystocia → NICU'
        },
        'ivf_pregnancy': {
            'probability': 15,
            'impact': 'NICU',
            'reason': 'Higher rates of prematurity and multiples'
        },
        'preterm': {
            'probability': 60,
            'impact': 'NICU/HIE/death',
            'reason': 'Immature lungs/organs increase need for NICU and complications'
        }
    }
    
    MODEL_DIR = 'ml_models'
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = None
        self._load_model()
    
    def _load_model(self):
        """Load trained model, scaler, and metadata"""
        try:
            model_path = os.path.join(self.MODEL_DIR, 'neonatal_model.pkl')
            scaler_path = os.path.join(self.MODEL_DIR, 'neonatal_scaler.pkl')
            metadata_path = os.path.join(self.MODEL_DIR, 'model_metadata.json')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.metadata = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load model - {str(e)}")
    
    def predict_patient_by_identifier(self, patient_id=None, file_number=None):
        """
        Predict neonatal complications for a patient
        
        Args:
            patient_id: Patient ID
            file_number: File number
        
        Returns:
            dict: Prediction result
        """
        try:
            from patients.models import Patient
            
            # Fetch patient
            if patient_id:
                patient = Patient.objects.get(patient_id=patient_id)
            elif file_number:
                patient = Patient.objects.get(file_number=file_number)
            else:
                return self._error_response('Either patient_id or file_number required')
            
            # Try direct rules first
            direct_result = self._check_direct_rules(patient)
            if direct_result['risk_factors']:
                # Aggregate probabilities and impacts
                max_prob = max(direct_result['probabilities'])
                avg_prob = int(sum(direct_result['probabilities']) / len(direct_result['probabilities']))
                impacts_dict = self._aggregate_impacts(direct_result)
                
                return self._format_response(
                    patient=patient,
                    probability=max_prob,
                    avg_probability=avg_prob,
                    method='direct_rule',
                    reasons=direct_result['all_reasons'],
                    impacts=impacts_dict,
                    risk_factors=direct_result['risk_factors'],
                    confidence=self._get_confidence(max_prob)
                )
            
            # Fall back to ML model
            if self.model and self.scaler and self.metadata:
                features = self._extract_features(patient)
                if not features:
                    return self._error_response('Insufficient patient data for prediction')
                
                probability, model_details = self._predict_ml(features)
                return self._format_response(
                    patient=patient,
                    probability=probability,
                    method='ml_model',
                    model_details=model_details,
                    confidence=self._get_confidence(probability)
                )
            
            return self._error_response('No prediction method available. Please train model first.')
            
        except Patient.DoesNotExist:
            return self._error_response(f'Patient not found')
        except Exception as e:
            return self._error_response(f'Prediction error: {str(e)}')
    
    def _check_direct_rules(self, patient):
        """Check for direct rule-based risk factors"""
        results = {
            'risk_factors': [],
            'all_reasons': [],
            'all_impacts': [],
            'probabilities': []
        }
        
        # Check CTG category
        if patient.ctg_category == 'category_iii_pathological':
            self._add_risk_factor(results, 'category_iii_pathological')
        elif patient.ctg_category == 'category_ii_suspicious':
            self._add_risk_factor(results, 'category_ii_suspicious')
        
        # Check placental issues
        current_pregnancy_maternal = patient.current_pregnancy_menternal or []
        
        if self._contains_keyword(current_pregnancy_maternal, 'Placental abruption'):
            self._add_risk_factor(results, 'placenta_abruption')
        if self._contains_keyword(current_pregnancy_maternal, 'Placenta previa'):
            self._add_risk_factor(results, 'placenta_previa')
        
        # Check multiple gestation
        if self._contains_keyword(current_pregnancy_maternal, 'Multiple gestation'):
            self._add_risk_factor(results, 'multiple_gestation')
        
        # Check presentation
        if patient.presentation and patient.presentation not in ['cephlic', None, '']:
            self._add_risk_factor(results, 'non_cephalic')
        
        # Check rupture duration
        if patient.rupture_duration_hour in ['18_24_hours', 'more_than_24_hours_prolonged_rupture']:
            self._add_risk_factor(results, 'prolonged_rupture')
        
        # Check PROM
        if patient.indication_of_induction == 'prelabor_rupture_of_membranes_prom':
            self._add_risk_factor(results, 'prom')
        
        # Check preeclampsia
        if self._contains_keyword(current_pregnancy_maternal, 'Pre-eclampsia'):
            self._add_risk_factor(results, 'preeclampsia')
        
        # Check diabetes
        menternal_medical = patient.menternal_medical or []
        if self._contains_keyword(menternal_medical, 'Diabetes'):
            self._add_risk_factor(results, 'diabetes')
        
        # Check severe anemia
        hb = patient.hb_g_dl
        if hb and hb < 7:
            self._add_risk_factor(results, 'severe_anemia')
        if self._contains_keyword(current_pregnancy_maternal, 'Severe anemia'):
            self._add_risk_factor(results, 'severe_anemia')
        
        # Check polyhydramnios
        if patient.liquor == 'polihydraminos':
            self._add_risk_factor(results, 'polyhydramios')
        
        # Check fetal weight
        if patient.estimated_fetal_weight_by_gm and patient.estimated_fetal_weight_by_gm >= 4000:
            self._add_risk_factor(results, 'macrosomia')
        
        # Check IVF
        if self._contains_keyword(current_pregnancy_maternal, 'IVF'):
            self._add_risk_factor(results, 'ivf_pregnancy')
        
        # Check preterm
        current_pregnancy_fetal = patient.current_pregnancy_fetal or []
        if self._contains_keyword(current_pregnancy_fetal, 'Preterm'):
            self._add_risk_factor(results, 'preterm')
        
        return results
    
    def _add_risk_factor(self, results, factor_key):
        """Add a risk factor to results"""
        if factor_key in self.DIRECT_RULES:
            rule = self.DIRECT_RULES[factor_key]
            results['risk_factors'].append(factor_key)
            results['probabilities'].append(rule['probability'])
            results['all_reasons'].append(rule['reason'])
            results['all_impacts'].append(rule['impact'])
    
    def _aggregate_impacts(self, direct_result):
        """Aggregate impacts by type"""
        impacts = {}
        for i, factor in enumerate(direct_result['risk_factors']):
            impact = direct_result['all_impacts'][i]
            prob = direct_result['probabilities'][i]
            
            if impact not in impacts:
                impacts[impact] = {
                    'probability': prob,
                    'factors': [factor]
                }
            else:
                impacts[impact]['factors'].append(factor)
                impacts[impact]['probability'] = max(impacts[impact]['probability'], prob)
        
        return impacts
    
    def _extract_features(self, patient):
        """Extract features from patient for ML prediction"""
        try:
            features = {}
            
            features['age'] = float(patient.age or 0)
            features['bmi'] = float(patient.bmi or 0)
            features['pulse'] = float(patient.pulse or 0)
            features['temp'] = float(patient.temp or 0)
            features['oxygen_sat'] = float(patient.oxygen_sat or 0)
            features['gravidity'] = float(patient.gravidity or 0)
            features['parity'] = float(patient.parity or 0)
            features['gestational_age'] = self._parse_gestational_age(patient.gestational_age)
            
            menternal_medical = patient.menternal_medical or []
            obstetric_history = patient.obstetric_history or []
            current_pregnancy_maternal = patient.current_pregnancy_menternal or []
            current_pregnancy_fetal = patient.current_pregnancy_fetal or []
            social = patient.social or []
            
            features['chronic_hypertension'] = 1 if self._contains_keyword(menternal_medical, 'Chronic hypertension') else 0
            features['diabetes'] = 1 if self._contains_keyword(menternal_medical, 'Diabetes') else 0
            features['multiple_cs'] = 1 if self._contains_keyword(obstetric_history, 'Multiple c-sections') else 0
            features['grand_multipara'] = 1 if self._contains_keyword(social, 'Grand multipara') else 0
            features['placenta_abruption'] = 1 if self._contains_keyword(current_pregnancy_maternal, 'Placental abruption') else 0
            features['placenta_previa'] = 1 if self._contains_keyword(current_pregnancy_maternal, 'Placenta previa') else 0
            features['multiple_gestation'] = 1 if self._contains_keyword(current_pregnancy_maternal, 'Multiple gestation') else 0
            features['preeclampsia'] = 1 if self._contains_keyword(current_pregnancy_maternal, 'Pre-eclampsia') else 0
            features['severe_anemia'] = 1 if self._contains_keyword(current_pregnancy_maternal, 'Severe anemia') else 0
            features['ivf_pregnancy'] = 1 if self._contains_keyword(current_pregnancy_maternal, 'IVF') else 0
            features['polyhydramios'] = 1 if patient.liquor == 'polihydraminos' else 0
            features['non_cephalic'] = 1 if patient.presentation not in ['cephlic', None, ''] else 0
            features['iugr'] = 1 if self._contains_keyword(current_pregnancy_fetal, 'IUGR') else 0
            features['preterm'] = 1 if self._contains_keyword(current_pregnancy_fetal, 'Preterm') else 0
            features['fetal_weight'] = float(patient.estimated_fetal_weight_by_gm or 0)
            features['ctg_abnormal'] = 1 if patient.ctg_category in ['category_ii_suspicious', 'category_iii_pathological'] else 0
            features['labor_duration'] = float(patient.labor_duration_hours or 0)
            features['prom'] = 1 if patient.indication_of_induction == 'prelabor_rupture_of_membranes_prom' else 0
            features['prolonged_rupture'] = 1 if patient.rupture_duration_hour in ['18_24_hours', 'more_than_24_hours_prolonged_rupture'] else 0
            features['placenta_location_lower'] = 1 if patient.placenta_location == 'lower' else 0
            features['cs_delivery'] = 1 if patient.mode_of_delivery == 'cs' else 0
            features['emergency_cs'] = 1 if patient.type_of_cs == 'emergency' else 0
            features['hb'] = float(patient.hb_g_dl or 7.5)
            features['platelets'] = float(patient.platelets_x10e9l or 150)
            
            if not any(features.values()):
                return None
            
            return features
            
        except Exception:
            return None
    
    def _predict_ml(self, features):
        """Get prediction from ML model"""
        try:
            feature_order = self.metadata['feature_names']
            X = np.array([[features.get(f, 0) for f in feature_order]])
            X_scaled = self.scaler.transform(X)
            probability = self.model.predict_proba(X_scaled)[0][1]
            
            model_details = {
                'model_type': 'RandomForest',
                'trained_samples': self.metadata.get('training_samples', 'N/A'),
                'test_samples': self.metadata.get('test_samples', 'N/A'),
                'trained_date': self.metadata.get('trained_date', 'N/A')
            }
            
            return int(probability * 100), model_details
        except Exception as e:
            print(f"ML prediction error: {str(e)}")
            return 0, None
    
    def _parse_gestational_age(self, ga_str):
        """Parse gestational age string to weeks"""
        try:
            if not ga_str:
                return 0
            parts = str(ga_str).split()
            return float(parts[0]) if parts else 0
        except:
            return 0
    
    def _contains_keyword(self, lst, keyword):
        """Check if list contains keyword"""
        if not isinstance(lst, list):
            return False
        return any(keyword.lower() in str(item).lower() for item in lst)
    
    def _get_confidence(self, probability):
        """Determine confidence level based on probability"""
        if probability >= 70:
            return 'high'
        elif probability >= 40:
            return 'moderate'
        else:
            return 'low'
    
    def _format_response(self, patient, probability, method, reasons=None, 
                        impacts=None, risk_factors=None, confidence=None, 
                        model_details=None, avg_probability=None):
        """Format prediction response"""
        return {
            'success': True,
            'patient_id': patient.patient_id,
            'file_number': patient.file_number,
            'patient_name': patient.name,
            'neonatal_complication_probability': probability,
            'average_probability': avg_probability or probability,
            'prediction_method': method,
            'reason': '; '.join(reasons) if reasons else 'No specific risk factors identified',
            'confidence': confidence or 'low',
            'risk_factors': risk_factors or [],
            'all_reasons': reasons or [],
            'neonatal_impacts': impacts or {},
            'model_details': model_details,
            'error': None
        }
    
    def _error_response(self, error_msg):
        """Format error response"""
        return {
            'success': False,
            'patient_id': None,
            'file_number': None,
            'patient_name': None,
            'neonatal_complication_probability': None,
            'average_probability': None,
            'prediction_method': None,
            'reason': None,
            'confidence': None,
            'risk_factors': [],
            'all_reasons': [],
            'neonatal_impacts': {},
            'model_details': None,
            'error': error_msg
        }







# File: your_app/views/neonatal_views.py

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
from django.core.management import call_command
from django.apps import apps
from io import StringIO
import json


class NeonatalPredictionViewSet(viewsets.ViewSet):
    """ViewSet for neonatal complication predictions"""
    
    permission_classes = [IsAuthenticated]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor = NeonatalPredictionService()
    
    @action(detail=False, methods=['post'])
    def predict(self, request):
        """
        Predict neonatal complications for a patient
        
        Request body:
        {
            "patient_id": "12345" or "file_number": "FILE001"
        }
        """
        try:
            data = request.data
            patient_id = data.get('patient_id')
            file_number = data.get('file_number')
            
            if not patient_id and not file_number:
                return Response(
                    {
                        'success': False,
                        'error': 'Either patient_id or file_number is required'
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            result = self.predictor.predict_patient_by_identifier(
                patient_id=patient_id,
                file_number=file_number
            )
            
            http_status = status.HTTP_200_OK if result['success'] else status.HTTP_400_BAD_REQUEST
            return Response(result, status=http_status)
            
        except Exception as e:
            return Response(
                {
                    'success': False,
                    'error': f'Prediction failed: {str(e)}'
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['post'])
    def predict_by_id(self, request):
        """
        Predict neonatal complications by patient ID
        
        Request body:
        {
            "patient_id": "12345"
        }
        """
        try:
            patient_id = request.data.get('patient_id')
            
            if not patient_id:
                return Response(
                    {
                        'success': False,
                        'error': 'patient_id is required'
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            result = self.predictor.predict_patient_by_identifier(patient_id=patient_id)
            
            http_status = status.HTTP_200_OK if result['success'] else status.HTTP_400_BAD_REQUEST
            return Response(result, status=http_status)
            
        except Exception as e:
            return Response(
                {
                    'success': False,
                    'error': f'Prediction failed: {str(e)}'
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['post'])
    def predict_by_file(self, request):
        """
        Predict neonatal complications by file number
        
        Request body:
        {
            "file_number": "FILE001"
        }
        """
        try:
            file_number = request.data.get('file_number')
            
            if not file_number:
                return Response(
                    {
                        'success': False,
                        'error': 'file_number is required'
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            result = self.predictor.predict_patient_by_identifier(file_number=file_number)
            
            http_status = status.HTTP_200_OK if result['success'] else status.HTTP_400_BAD_REQUEST
            return Response(result, status=http_status)
            
        except Exception as e:
            return Response(
                {
                    'success': False,
                    'error': f'Prediction failed: {str(e)}'
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    


from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated

@api_view(['GET'])
@permission_classes([AllowAny])
def predict_neonatal_by_identifier(request):
    """
    API endpoint to predict neonatal complications
    
    POST /api/neonatal/predict/
    
    Request body:
    {
        "patient_id": "12345" (or "file_number": "FILE001")
    }
    
    Response:
    {
        "success": true,
        "patient_id": "12345",
        "file_number": "FILE001",
        "patient_name": "John Doe",
        "neonatal_complication_probability": 45,
        "average_probability": 45,
        "prediction_method": "direct_rule",
        "reason": "Multiple risk factors identified...",
        "confidence": "moderate",
        "risk_factors": ["multiple_gestation", "preeclampsia"],
        "all_reasons": ["Reason 1", "Reason 2"],
        "neonatal_impacts": {
            "NICU": {"probability": 45, "factors": ["factor1"]},
            "HIE": {"probability": 25, "factors": ["factor2"]}
        },
        "model_details": null,
        "error": null
    }
    """
    try:
        file_number = request.GET.get('file_number')
        
        if not file_number:
            return Response(
                {
                    'success': False,
                    'error': 'Either patient_id or file_number is required'
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        predictor = NeonatalPredictionService()
        result = predictor.predict_patient_by_identifier(
            patient_id=None,
            file_number=file_number
        )
        
        http_status = status.HTTP_200_OK if result['success'] else status.HTTP_400_BAD_REQUEST
        return Response(result, status=http_status)
        
    except Exception as e:
        return Response(
            {
                'success': False,
                'patient_id': None,
                'file_number': None,
                'patient_name': None,
                'neonatal_complication_probability': None,
                'average_probability': None,
                'prediction_method': None,
                'reason': None,
                'confidence': None,
                'risk_factors': [],
                'all_reasons': [],
                'neonatal_impacts': {},
                'model_details': None,
                'error': str(e)
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
