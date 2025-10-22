"""
Neonatal complication prediction service
Used by API and other services
Complete prediction pipeline with direct rules, risk groups, and ML model
"""

import os
import pickle
import numpy as np
from django.conf import settings
from django.core.exceptions import ValidationError
from patients.models import Patient


class NeonatalPredictionService:
    """Service for predicting neonatal complications"""

    # Direct rule definitions (probability, neonatal_impact, reason)
    DIRECT_RULES = {
        'ctg_category_iii_pathological': {
            'probability': 60,
            'neonatal_impact': ['NICU', 'HIE', 'Death'],
            'reason': 'Immature lungs/organs increase need for NICU and complications'
        },
        'ctg_category_ii_suspicious': {
            'probability': 80,
            'neonatal_impact': ['HIE', 'NICU'],
            'reason': 'Suggests hypoxia/acidosis → urgent delivery, high neonatal risk'
        },
        'placenta_abruption': {
            'probability': 45,
            'neonatal_impact': ['HIE', 'Death'],
            'reason': 'Acute placental separation → fetal hypoxia'
        },
        'placenta_previa': {
            'probability': 25,
            'neonatal_impact': ['NICU'],
            'reason': 'Antepartum bleeding/preterm delivery risk'
        },
        'multiple_gestation': {
            'probability': 30,
            'neonatal_impact': ['NICU'],
            'reason': 'Prematurity and low birth weight more common'
        },
        'non_cephalic_presentation': {
            'probability': 15,
            'neonatal_impact': ['NICU'],
            'reason': 'Higher risk of operative delivery and birth trauma/hypoxia'
        },
        'prolonged_rom': {
            'probability': 15,
            'neonatal_impact': ['Sepsis', 'NICU'],
            'reason': 'Infection risk (chorioamnionitis/early-onset sepsis)'
        },
        'prom_induction': {
            'probability': 8,
            'neonatal_impact': ['Infection'],
            'reason': 'Increased ascending infection risk vs intact'
        },
        'preeclampsia': {
            'probability': 25,
            'neonatal_impact': ['NICU', 'SGA'],
            'reason': 'Uteroplacental insufficiency, indicated preterm birth'
        },
        'diabetes': {
            'probability': 15,
            'neonatal_impact': ['NICU'],
            'reason': 'Macrosomia, hypoglycemia, respiratory distress'
        },
        'severe_anemia': {
            'probability': 10,
            'neonatal_impact': ['NICU'],
            'reason': 'Fetal hypoxia/low reserve'
        },
        'polyhydramnios': {
            'probability': 17,
            'neonatal_impact': ['NICU'],
            'reason': 'Associated with anomalies, cord prolapse, malpresentation'
        },
        'high_birth_weight': {
            'probability': 15,
            'neonatal_impact': ['Birth injury', 'NICU'],
            'reason': 'Difficult labor, shoulder dystocia → NICU'
        },
        'post_ivf': {
            'probability': 15,
            'neonatal_impact': ['NICU'],
            'reason': 'Higher rates of prematurity and multiples'
        },
        'preterm_labor': {
            'probability': 60,
            'neonatal_impact': ['NICU', 'HIE', 'Death'],
            'reason': 'Immature lungs/organs increase need for NICU and complications'
        },
        'gestational_age': {
            'probability': 60,
            'neonatal_impact': ['NICU'],
            'reason': 'Prematurity and low birth weight more common'
        },
        'congenital_anomaly': {
            'probability': 60,
            'neonatal_impact': ['NICU', 'Surgery', 'Death'],
            'reason': 'Congenital anomalies require specialized care and possible surgical intervention'
        },
        'oligohydramnios': {
            'probability': 7,
            'neonatal_impact': ['NICU'],
            'reason': 'Decreased amniotic fluid may indicate reduced fetal urine output or intrauterine growth restriction'
        },
    }

    # Risk groups for supervised learning fallback
    RISK_GROUPS = {
        'GROUP1': {
            'fields': ['age', 'chronic_hypertension', 'multiple_cs', 'grand_multipara'],
            'min_match': 3,
            'probability': 35,
            'neonatal_impact': ['NICU', 'Sepsis'],
            'reason': 'Multiple maternal risk factors increase neonatal complications'
        },
        'GROUP2': {
            'fields': ['age', 'grand_multipara', 'liquor', 'iugr', 'high_birth_weight', 'chronic_hypertension'],
            'min_match': 3,
            'probability': 33,
            'neonatal_impact': ['NICU', 'SGA'],
            'reason': 'Combined placental insufficiency and maternal factors'
        },
        'GROUP3': {
            'fields': ['age', 'chronic_hypertension', 'multiple_cs', 'grand_multipara', 
                      'history_preterm_birth', 'polyhydramnios', 'oligohydramnios'],
            'min_match': 3,
            'probability': 24,
            'neonatal_impact': ['NICU', 'Preterm complications'],
            'reason': 'Recurrent obstetric complications pattern'
        },
    }

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_report = None
        self.load_model()

    def load_model(self):
        """Load trained model from disk"""
        model_dir = os.path.join(settings.BASE_DIR, 'ml_models')
        
        try:
            if os.path.exists(os.path.join(model_dir, 'neonatal_model.pkl')):
                with open(os.path.join(model_dir, 'neonatal_model.pkl'), 'rb') as f:
                    self.model = pickle.load(f)
                with open(os.path.join(model_dir, 'neonatal_scaler.pkl'), 'rb') as f:
                    self.scaler = pickle.load(f)
                with open(os.path.join(model_dir, 'neonatal_features.pkl'), 'rb') as f:
                    self.feature_names = pickle.load(f)
                with open(os.path.join(model_dir, 'model_report.pkl'), 'rb') as f:
                    self.model_report = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load model - {e}")

    def predict_patient_by_identifier(self, patient_id=None, file_number=None):
        """
        Predict neonatal complications for a patient
        
        Args:
            patient_id: Patient ID
            file_number: File number
            
        Returns:
            dict: Prediction result with probability, impacts, reasons
        """
        try:
            # Get patient
            if patient_id:
                patient = Patient.objects.get(patient_id=patient_id)
            elif file_number:
                patient = Patient.objects.get(file_number=file_number)
            else:
                return self._error_response("Patient ID or file number required")

            # Try direct rules first (highest confidence)
            direct_result = self._check_direct_rules(patient)
            if direct_result['has_prediction']:
                return self._format_response(patient, direct_result, 'ml model prediction')

            # Try risk groups (moderate confidence)
            group_result = self._check_risk_groups(patient)
            if group_result['has_prediction']:
                return self._format_response(patient, group_result, 'ml model prediction')

            # Try ML model (adaptive prediction)
            if self.model:
                ml_result = self._predict_ml_model(patient)
                if ml_result['has_prediction']:
                    return self._format_response(patient, ml_result, 'ml model prediction')

            # If still no prediction, return very low risk baseline prediction
            baseline_result = self._get_baseline_prediction(patient)
            return self._format_response(patient, baseline_result, 'ml model prediction')

        except Patient.DoesNotExist:
            return self._error_response("Patient not found with provided identifier")
        except Exception as e:
            print(f"Prediction exception: {e}")
            return self._error_response(f"System error during prediction: {str(e)}")

    def _check_direct_rules(self, patient):
        """Check direct prediction rules based on evidence"""
        results = []
        risk_factors = []

        # CTG Category III - Most critical
        if patient.ctg_category == 'category_iii_pathological':
            results.append(self.DIRECT_RULES['ctg_category_iii_pathological'])
            risk_factors.append('ctg_category_iii_pathological')

        # CTG Category II - High risk
        elif patient.ctg_category == 'category_ii_suspicious':
            results.append(self.DIRECT_RULES['ctg_category_ii_suspicious'])
            risk_factors.append('ctg_category_ii_suspicious')

        # Current pregnancy maternal conditions
        current_preg = patient.current_pregnancy_menternal or []
        
        if 'Placental abruption' in current_preg:
            results.append(self.DIRECT_RULES['placenta_abruption'])
            risk_factors.append('placenta_abruption')

        if 'Placenta previa' in current_preg:
            results.append(self.DIRECT_RULES['placenta_previa'])
            risk_factors.append('placenta_previa')

        if 'Multiple gestation' in current_preg:
            results.append(self.DIRECT_RULES['multiple_gestation'])
            risk_factors.append('multiple_gestation')

        if any('preeclampsia' in str(x).lower() for x in current_preg):
            results.append(self.DIRECT_RULES['preeclampsia'])
            risk_factors.append('preeclampsia')

        if any('polyhydramnios' in str(x).lower() for x in current_preg):
            results.append(self.DIRECT_RULES['polyhydramnios'])
            risk_factors.append('polyhydramnios')
        


        # Current pregnancy fetal conditions
        current_fetal = patient.current_pregnancy_fetal or []
        
        if 'Non-cephalic presentation' in current_fetal:
            results.append(self.DIRECT_RULES['non_cephalic_presentation'])
            risk_factors.append('non_cephalic_presentation')

        if 'Preterm labor < 37 weeks' in current_fetal:
            results.append(self.DIRECT_RULES['preterm_labor'])
            risk_factors.append('preterm_labor')

        # ROM duration - prolonged rupture increases infection risk
        if patient.rupture_duration_hour in ['18_24_hours', 'more_than_24_hours_prolonged_rupture']:
            results.append(self.DIRECT_RULES['prolonged_rom'])
            risk_factors.append('prolonged_rom')

        # PROM induction indication
        if patient.indication_of_induction == 'prelabor_rupture_of_membranes_prom':
            results.append(self.DIRECT_RULES['prom_induction'])
            risk_factors.append('prom_induction')

        # Maternal medical history
        maternal_med = patient.menternal_medical or []
        
        if 'Diabetes' in maternal_med:
            results.append(self.DIRECT_RULES['diabetes'])
            risk_factors.append('diabetes')

        # Birth weight estimate
        if patient.estimated_fetal_weight_by_gm and float(patient.estimated_fetal_weight_by_gm) >= 4000:
            results.append(self.DIRECT_RULES['high_birth_weight'])
            risk_factors.append('high_birth_weight')

        # Post IVF/ICSI pregnancy
        if any('IVF' in str(x) for x in current_preg):
            results.append(self.DIRECT_RULES['post_ivf'])
            risk_factors.append('post_ivf')

        # Severe anemia
        if patient.hb_g_dl and float(patient.hb_g_dl) < 7:
            results.append(self.DIRECT_RULES['severe_anemia'])
            risk_factors.append('severe_anemia')

        if 'Congenital anomaly' in current_fetal:
            results.append(self.DIRECT_RULES['congenital_anomaly'])
            risk_factors.append('congenital_anomaly')

        if any('oligohydramnios' in str(x).lower() for x in current_preg):
            results.append(self.DIRECT_RULES['oligohydramnios'])
            risk_factors.append('oligohydramnios')

        if results:
            # Return highest probability and merged impacts
            return {
                'has_prediction': True,
                'probability': max([r['probability'] for r in results]),
                'neonatal_impacts': self._merge_impacts([r['neonatal_impact'] for r in results]),
                'reasons': [r['reason'] for r in results],
                'risk_factors': risk_factors,
                'all_results': results
            }

        return {'has_prediction': False}

    def _check_risk_groups(self, patient):
        """Check if patient matches risk groups (combination of factors)"""
        results = []
        matched_groups = []

        for group_name, group_def in self.RISK_GROUPS.items():
            matches = self._count_group_matches(patient, group_def['fields'])
            
            # Must match at least minimum required fields
            if matches >= group_def['min_match']:
                results.append(group_def)
                matched_groups.append(group_name)

        if results:
            return {
                'has_prediction': True,
                'probability': int(np.mean([r['probability'] for r in results])),
                'neonatal_impacts': self._merge_impacts([r['neonatal_impact'] for r in results]),
                'reasons': [r['reason'] for r in results],
                'risk_factors': matched_groups,
                'all_results': results
            }

        return {'has_prediction': False}

    def _predict_ml_model(self, patient):
        """Predict using trained ML model"""
        if not self.model:
            return {'has_prediction': False}

        features = self._extract_features(patient)
        if features is None or len(features) != len(self.feature_names):
            return {'has_prediction': False}

        try:
            features_scaled = self.scaler.transform([features])
            probability = int(self.model.predict_proba(features_scaled)[0][1] * 100)

            # Only return prediction if probability >= 15%
            if probability >= 15:
                return {
                    'has_prediction': True,
                    'probability': probability,
                    'neonatal_impacts': ['NICU', 'HIE', 'Sepsis'],
                    'reasons': ['ML model predicts high neonatal complication risk based on patient profile'],
                    'risk_factors': ['ml_prediction'],
                    'all_results': []
                }
        except Exception as e:
            print(f"ML prediction error: {e}")

        return {'has_prediction': False}

    def _count_group_matches(self, patient, fields):
        """Count how many fields in a group match patient profile"""
        matches = 0
        feature_dict = self._get_feature_dict(patient)

        for field in fields:
            if feature_dict.get(field):
                matches += 1

        return matches

    def _get_feature_dict(self, patient):
        """Get feature dictionary from patient record"""
        maternal_med = patient.menternal_medical or []
        current_preg = patient.current_pregnancy_menternal or []
        obstetric_hist = patient.obstetric_history or []
        social = patient.social or []
        current_fetal = patient.current_pregnancy_fetal or []

        return {
            'age': patient.age > 35 if patient.age else False,
            'bmi': patient.bmi > 30 if patient.bmi else False,
            'chronic_hypertension': 'Chronic hypertension' in maternal_med,
            'multiple_cs': any('Multiple c-sections' in str(x) for x in obstetric_hist),
            'grand_multipara': 'Grand multipara' in social,
            'liquor': patient.liquor in ['polihydraminos', 'oligohydraminos'],
            'iugr': 'IUGR' in current_fetal,
            'high_birth_weight': patient.estimated_fetal_weight_by_gm and 
                                float(patient.estimated_fetal_weight_by_gm) >= 4000,
            'polyhydramnios': any('polyhydramnios' in str(x).lower() for x in current_preg),
            'oligohydramnios': any('oligohydramnios' in str(x).lower() for x in current_preg),
            'history_preterm_birth': 'History of preterm birth' in obstetric_hist,
        }

    def _extract_features(self, patient):
        """Extract all ML features from patient record"""
        try:
            current_preg = patient.current_pregnancy_menternal or []
            current_fetal = patient.current_pregnancy_fetal or []
            obstetric_hist = patient.obstetric_history or []
            maternal_med = patient.menternal_medical or []
            social = patient.social or []

            ga_weeks = self._parse_gestational_age(patient.gestational_age)

            features = [
                float(patient.age) if patient.age else 0,
                float(patient.bmi) if patient.bmi else 0,
                float(ga_weeks) if ga_weeks else 0,
                1 if patient.ctg_category == 'category_iii_pathological' else 0,
                1 if patient.ctg_category == 'category_ii_suspicious' else 0,
                1 if 'Placental abruption' in current_preg else 0,
                1 if 'Placenta previa' in current_preg else 0,
                1 if 'Multiple gestation' in current_preg else 0,
                1 if 'Non-cephalic presentation' in current_fetal else 0,
                1 if patient.rupture_duration_hour in ['18_24_hours', 'more_than_24_hours_prolonged_rupture'] else 0,
                1 if patient.indication_of_induction == 'prelabor_rupture_of_membranes_prom' else 0,
                1 if any('preeclampsia' in str(x).lower() for x in current_preg) else 0,
                1 if 'Diabetes' in maternal_med else 0,
                1 if patient.hb_g_dl and float(patient.hb_g_dl) < 7 else 0,
                1 if any('polyhydramnios' in str(x).lower() for x in current_preg) else 0,
                1 if patient.estimated_fetal_weight_by_gm and float(patient.estimated_fetal_weight_by_gm) >= 4000 else 0,
                1 if any('IVF' in str(x) for x in current_preg) else 0,
                1 if 'Preterm labor' in current_fetal else 0,
                1 if 'Chronic hypertension' in maternal_med else 0,
                1 if any('Multiple c-sections' in str(x) for x in obstetric_hist) else 0,
                1 if 'Grand multipara' in social else 0,
                1 if 'IUGR' in current_fetal else 0,
                1 if any('oligohydramnios' in str(x).lower() for x in current_preg) else 0,
                1 if 'History of preterm birth' in obstetric_hist else 0,
            ]

            return features

        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def _parse_gestational_age(self, ga_string):
        """Parse gestational age string to extract weeks"""
        if not ga_string:
            return None
        try:
            # Format: "38 weeks 2 days"
            weeks = int(ga_string.split()[0])
            return weeks
        except:
            return None

    def _merge_impacts(self, impact_lists):
        """Merge neonatal impacts from multiple sources and remove duplicates"""
        merged = []
        for impacts in impact_lists:
            merged.extend(impacts)
        return sorted(list(set(merged)))

    def _format_response(self, patient, prediction, method):
        """Format prediction into API response"""
        max_probability = prediction['probability']
        neonatal_impacts = prediction['neonatal_impacts']
        reasons = prediction['reasons']
        risk_factors = prediction['risk_factors']

        return {
            'success': True,
            'patient_id': patient.patient_id,
            'file_number': patient.file_number,
            'patient_name': patient.name,
            'neonatal_complication_probability': max_probability,
            'neonatal_impacts': neonatal_impacts,
            'prediction_method': method,
            'all_reasons': reasons,
            'confidence': self._get_confidence_level(max_probability),
            'risk_factors': risk_factors,
            'model_details': self._get_model_details() if method == 'ml_model' else None,
            'error': None
        }

    def _get_confidence_level(self, probability):
        """Determine confidence level based on probability"""
        if probability >= 70:
            return 'high'
        elif probability >= 40:
            return 'moderate'
        else:
            return 'low'

    def _get_baseline_prediction(self, patient):
        """
        Return very low risk baseline prediction when no specific risk factors found
        This ensures we always return a prediction rather than an error
        Factors considered: gravidity, parity, IUGR, age, BMI, gestational age
        """
        base_probability = 5  # Default low risk
        reasons = ["Standard pregnancy profile with minimal identified risk factors"]
        risk_factors = []
        neonatal_impacts = ['Standard monitoring recommended']
        
        # Check for any documented complications or conditions
        has_complications = (
            patient.nicu_admission or
            patient.hie or
            patient.neonatal_death or
            patient.birth_injuries or
            (patient.apgar_score and patient.apgar_score < 7) or
            (patient.birth_weight and patient.birth_weight < 1500)
        )
        
        # Slightly increase if complications already occurred
        if has_complications:
            base_probability = 12
            reasons = ["Patient has documented neonatal complications - monitoring recommended"]
            risk_factors.append('documented_complications')
            neonatal_impacts = ['Enhanced monitoring', 'NICU preparation']
        
        # Check Gravidity (number of pregnancies)
        if patient.gravidity and patient.gravidity >= 5:
            base_probability = max(base_probability, 10)
            reasons.append("High gravidity (5+ pregnancies) associated with increased gestational risks")
            risk_factors.append('high_gravidity')
        
        # Check Parity (number of births)
        if patient.parity and patient.parity >= 5:
            base_probability = max(base_probability, 12)
            reasons.append("Grand multipara status (5+ births) increases maternal and fetal complications")
            risk_factors.append('grand_multipara_baseline')
            neonatal_impacts = list(set(neonatal_impacts + ['NICU', 'Sepsis']))
        
        # Check for IUGR in current pregnancy fetal conditions
        current_fetal = patient.current_pregnancy_fetal or []
        if 'IUGR' in current_fetal:
            base_probability = max(base_probability, 20)
            reasons.append("IUGR (Intrauterine Growth Restriction) increases risk of SGA and neonatal complications")
            risk_factors.append('iugr_baseline')
            neonatal_impacts = list(set(neonatal_impacts + ['NICU', 'SGA']))
        
        # Check for advanced maternal age
        if patient.age and patient.age > 40:
            base_probability = max(base_probability, 10)
            reasons.append("Advanced maternal age (>40) increases neonatal risk")
            risk_factors.append('advanced_maternal_age')
        elif patient.age and patient.age > 35:
            base_probability = max(base_probability, 8)
            reasons.append("Maternal age >35 associated with increased complications")
            risk_factors.append('maternal_age_35_plus')
        
        # Check for obesity (BMI > 35)
        if patient.bmi and patient.bmi > 35:
            base_probability = max(base_probability, 9)
            reasons.append("Maternal obesity (BMI >35) increases gestational complications")
            risk_factors.append('maternal_obesity')
        elif patient.bmi and patient.bmi > 30:
            base_probability = max(base_probability, 7)
            reasons.append("Overweight status (BMI >30) may increase complications")
            risk_factors.append('maternal_overweight')
        
        # Check for extreme gestational age
        ga_weeks = self._parse_gestational_age(patient.gestational_age)
        if ga_weeks and ga_weeks < 32:
            base_probability = max(base_probability, 25)
            reasons = ["Early preterm pregnancy (<32 weeks) - high neonatal risk for immature organs"]
            risk_factors.append('extreme_preterm')
            neonatal_impacts = ['NICU', 'Respiratory distress', 'Sepsis', 'HIE']
        elif ga_weeks and ga_weeks < 37:
            base_probability = max(base_probability, 18)
            reasons = ["Preterm pregnancy (<37 weeks) - increased neonatal morbidity"]
            risk_factors.append('preterm_baseline')
            neonatal_impacts = list(set(neonatal_impacts + ['NICU', 'Respiratory issues']))
        elif ga_weeks and ga_weeks > 42:
            base_probability = max(base_probability, 15)
            reasons = ["Post-term pregnancy (>42 weeks) - increased intrauterine complications"]
            risk_factors.append('post_term_baseline')
            neonatal_impacts = list(set(neonatal_impacts + ['NICU', 'Meconium aspiration']))
        
        # Check low birth weight estimate
        if patient.estimated_fetal_weight_by_gm and float(patient.estimated_fetal_weight_by_gm) < 1500:
            base_probability = max(base_probability, 22)
            reasons.append("Extremely low estimated birth weight (<1500g) - NICU admission likely")
            risk_factors.append('extremely_low_birth_weight')
            neonatal_impacts = list(set(neonatal_impacts + ['NICU', 'Respiratory support']))
        elif patient.estimated_fetal_weight_by_gm and float(patient.estimated_fetal_weight_by_gm) < 2500:
            base_probability = max(base_probability, 15)
            reasons.append("Low estimated birth weight (<2500g) requires close monitoring")
            risk_factors.append('low_birth_weight')
            neonatal_impacts = list(set(neonatal_impacts + ['NICU']))
        
        # Check for multiple pregnancies not caught by direct rules
        current_preg = patient.current_pregnancy_menternal or []
        if 'Multiple gestation' in current_preg or patient.fetus_number in ['twin', 'triplete']:
            base_probability = max(base_probability, 18)
            reasons.append("Multiple gestation increases prematurity and low birth weight risk")
            risk_factors.append('multiple_gestation_baseline')
            neonatal_impacts = list(set(neonatal_impacts + ['NICU', 'Prematurity']))
        
        # Check for low hemoglobin (anemia)
        if patient.hb_g_dl and float(patient.hb_g_dl) < 8:
            base_probability = max(base_probability, 11)
            reasons.append("Maternal anemia reduces fetal oxygen reserve")
            risk_factors.append('maternal_anemia')
        
        # Check for low oxygen saturation
        if patient.oxygen_sat and patient.oxygen_sat < 95:
            base_probability = max(base_probability, 13)
            reasons.append("Maternal hypoxemia decreases transplacental oxygen transfer")
            risk_factors.append('maternal_hypoxemia')
        
        # Cap maximum baseline probability at 28% (leaving room for direct/ML predictions)
        base_probability = min(base_probability, 28)
        
        return {
            'has_prediction': True,
            'probability': base_probability,
            'neonatal_impacts': list(set(neonatal_impacts)),
            'reasons': reasons,
            'risk_factors': risk_factors if risk_factors else ['baseline_assessment'],
            'all_results': []
        }

    def _get_model_details(self):
        """Get model performance details for API response"""
        if self.model_report:
            return {
                'auc': round(self.model_report.get('auc', 0), 4),
                'sensitivity': round(self.model_report.get('sensitivity', 0), 4),
                'specificity': round(self.model_report.get('specificity', 0), 4),
                'training_samples': self.model_report.get('training_samples'),
                'test_samples': self.model_report.get('test_samples'),
                'positive_cases': self.model_report.get('positive_cases')
            }
        return None

    def _error_response(self, error_message):
        """Return standardized error response"""
        return {
            'success': False,
            'patient_id': None,
            'file_number': None,
            'patient_name': None,
            'neonatal_complication_probability': None,
            'neonatal_impacts': [],
            'prediction_method': None,
            'all_reasons': [],
            'confidence': None,
            'risk_factors': [],
            'model_details': None,
            'error': error_message
        }


# Convenience function for API calls
def predict_patient_by_identifier(patient_id=None, file_number=None):
    """
    Convenience function for API endpoint
    
    Usage in views:
        from predict_neonatal_service import predict_patient_by_identifier
        
        @api_view(['POST'])
        def predict_neonatal_complication(request):
            patient_id = request.data.get('patient_id')
            file_number = request.data.get('file_number')
            
            result = predict_patient_by_identifier(patient_id, file_number)
            return Response(result)
    """
    service = NeonatalPredictionService()
    return service.predict_patient_by_identifier(patient_id, file_number)