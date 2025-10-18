"""
Django Management Command: train_pph_model.py
Location: yourapp/management/commands/train_pph_model.py

Usage: python manage.py train_pph_model
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from django.core.management.base import BaseCommand
from django.conf import settings
from patients.models import Patient

class Command(BaseCommand):
    help = 'Train PPH prediction model'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting PPH model training...'))
        
        # Create models directory if not exists
        models_dir = Path(settings.BASE_DIR) / 'ml_models'
        models_dir.mkdir(exist_ok=True)
        
        # Fetch all patients with sever_pph label
        patients = Patient.objects.all()
        
        if not patients.exists():
            self.stdout.write(self.style.ERROR('No patients found in database'))
            return
        
        # Prepare training data
        X, y, feature_names = self._prepare_data(patients)
        
        if len(X) == 0:
            self.stdout.write(self.style.ERROR('No valid training data'))
            return
        
        self.stdout.write(f'Training data shape: {X.shape}')
        self.stdout.write(f'Positive PPH cases: {sum(y)} ({sum(y)/len(y)*100:.1f}%)')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        auc = roc_auc_score(y_test, y_pred_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        self.stdout.write(self.style.SUCCESS(f'\n=== Model Performance ==='))
        self.stdout.write(f'AUC: {auc:.4f}')
        self.stdout.write(f'Sensitivity: {sensitivity:.4f}')
        self.stdout.write(f'Specificity: {specificity:.4f}')
        self.stdout.write(f'\nClassification Report:\n{classification_report(y_test, y_pred)}')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.stdout.write(f'\nTop 10 Important Features:')
        self.stdout.write(feature_importance.head(10).to_string())
        
        # Save model and scaler
        model_path = models_dir / 'pph_model.pkl'
        scaler_path = models_dir / 'pph_scaler.pkl'
        metrics_path = models_dir / 'pph_metrics.json'
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        metrics = {
            'auc': float(auc),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'feature_names': feature_names,
            'feature_importance': feature_importance.to_dict('records')
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('PPH Prediction - ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        roc_path = models_dir / 'pph_roc_curve.png'
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.stdout.write(self.style.SUCCESS(f'\nModel saved to {model_path}'))
        self.stdout.write(self.style.SUCCESS(f'Scaler saved to {scaler_path}'))
        self.stdout.write(self.style.SUCCESS(f'Metrics saved to {metrics_path}'))
        self.stdout.write(self.style.SUCCESS(f'ROC curve saved to {roc_path}'))

    def _prepare_data(self, patients):
        """Prepare training data from patients"""
        X_list = []
        y_list = []
        feature_names = []
        
        for patient in patients:
            features = self._extract_features(patient)
            if features is not None:
                X_list.append(list(features.values()))
                y_list.append(int(patient.sever_pph))
                feature_names = list(features.keys())
        
        return np.array(X_list), np.array(y_list), feature_names

    def _extract_features(self, patient):
        """Extract features from patient"""
        try:
            features = {}
            
            # Basic demographics
            features['age'] = patient.age or 0
            features['parity'] = patient.parity or 0
            features['bmi'] = float(patient.bmi) if patient.bmi else 0
            
            # Vital signs
            features['pulse'] = patient.pulse or 0
            try:
                bp_systolic = int(patient.bp.split('/')[0]) if patient.bp else 0
                features['bp_systolic'] = bp_systolic
            except:
                features['bp_systolic'] = 0
            
            features['hb'] = patient.hb_g_dl or 0
            features['platelets'] = patient.platelets_x10e9l or 0
            
            # Measurements
            features['cervical_dilatation_at_admission'] = patient.cervical_dilatation_at_admission or 0
            features['labor_duration_hours'] = patient.labor_duration_hours or 0
            features['estimated_fetal_weight'] = patient.estimated_fetal_weight_by_gm or 0
            
            # Medical history flags
            features['chronic_hypertension'] = self._check_in_json(patient.menternal_medical, ['Chronic hypertension'])
            features['diabetes'] = self._check_in_json(patient.menternal_medical, ['Diabetes', 'Gestational diabetes'])
            features['blood_transfusion_history'] = self._check_in_json(patient.menternal_medical, ['History of blood transfusion'])
            features['severe_anemia'] = self._check_in_json(patient.current_pregnancy_menternal, ['Severe anemia'])
            
            # Obstetric history
            features['prev_cs_single'] = self._check_in_json(patient.obstetric_history, ['Previous c-section'])
            features['prev_cs_multiple'] = self._check_in_json(patient.obstetric_history, ['Multiple c-sections'])
            features['prev_pph'] = self._check_in_json(patient.obstetric_history, ['History of postpartum hemorrhage'])
            features['prolonged_labor_history'] = self._check_in_json(patient.obstetric_history, ['Obstructed/prolonged labor'])
            features['prev_placenta_abruption'] = self._check_in_json(patient.obstetric_history, ['Placental abruption'])
            features['prev_placenta_previa'] = self._check_in_json(patient.obstetric_history, ['Placenta previa'])
            
            # Current pregnancy complications
            features['placenta_abruption_current'] = self._check_in_json(patient.current_pregnancy_menternal, ['Placental abruption'])
            features['placenta_previa_current'] = self._check_in_json(patient.current_pregnancy_menternal, ['Placenta previa'])
            features['preeclampsia'] = self._check_in_json(patient.current_pregnancy_menternal, ['Pre-eclampsia'])
            features['multiple_gestation'] = self._check_in_json(patient.current_pregnancy_menternal, ['Multiple gestation'])
            
            # Delivery details
            features['grand_multipara'] = 1 if patient.parity >= 5 else 0
            features['bmi_obese_35_39'] = 1 if 35 <= float(patient.bmi or 0) < 39.5 else 0
            features['bmi_obese_40_plus'] = 1 if float(patient.bmi or 0) >= 40 else 0
            features['fetus_number_twin'] = 1 if patient.fetus_number == 'twin' else 0
            features['fetus_number_triplet'] = 1 if patient.fetus_number == 'triplete' else 0
            
            # Delivery mode and complications
            features['instrumental_delivery'] = int(patient.instrumental_delivery)
            features['cs_emergency'] = 1 if patient.type_of_cs == 'emergency' else 0
            features['mode_cs'] = 1 if patient.mode_of_delivery == 'cs' else 0
            features['polyhydramnios'] = 1 if patient.liquor == 'polihydraminos' else 0
            
            # CTG
            features['ctg_category_ii'] = 1 if patient.ctg_category == 'category_ii_suspicious' else 0
            features['ctg_category_iii'] = 1 if patient.ctg_category == 'category_iii_pathological' else 0
            
            # Placenta location
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
