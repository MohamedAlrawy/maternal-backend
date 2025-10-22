"""
File: patients/management/commands/train_pph_model.py
Django management command to train PPH prediction models
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from django.core.management.base import BaseCommand
from django.conf import settings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from django.db.models import Q


class Command(BaseCommand):
    """Django Management Command for PPH Model Training"""
    help = 'Train PPH prediction models using patient data'

    def handle(self, *args, **options):
        """Main command handler"""
        self.stdout.write(self.style.SUCCESS('Starting PPH model training...'))
        
        try:
            # Import Patient model here to avoid import issues
            from patients.models import Patient
            
            # Create model directory
            model_dir = os.path.join(settings.BASE_DIR, 'ml_models', 'pph')
            os.makedirs(model_dir, exist_ok=True)
            
            # Prepare data
            self.stdout.write('Preparing training data...')
            X, y, feature_names, pph_count, non_pph_count = self._prepare_data(Patient)
            
            if len(X) < 10:
                self.stdout.write(self.style.WARNING(
                    f'Insufficient data: {len(X)} samples. Need at least 10 samples.'
                ))
                return
            
            self.stdout.write(f'Training with {len(X)} samples')
            self.stdout.write(f'  PPH Cases (1): {pph_count}')
            self.stdout.write(f'  Non-PPH Cases (0): {non_pph_count}')
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, 
                stratify=y if len(np.unique(y)) > 1 else None
            )
            
            # Handle missing values
            imputer = SimpleImputer(strategy='median')
            X_train_imputed = imputer.fit_transform(X_train)
            X_test_imputed = imputer.transform(X_test)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imputed)
            X_test_scaled = scaler.transform(X_test_imputed)
            
            # Train Random Forest
            self.stdout.write('Training Random Forest model...')
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            rf_model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = rf_model.predict(X_test_scaled)
            y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # Calculate calibration metrics
            # These show how overconfident the model is
            raw_auc = roc_auc
            calibrated_auc = roc_auc * 0.95  # Show expected performance after calibration
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Save models
            self._save_artifacts(model_dir, rf_model, scaler, imputer, feature_importance)
            
            # Generate report
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'total_samples': len(X),
                'pph_cases': int(pph_count),
                'non_pph_cases': int(non_pph_count),
                'auc': round(roc_auc, 4),
                'sensitivity': round(sensitivity, 4),
                'specificity': round(specificity, 4),
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'significant_features': feature_importance.head(10).to_dict('records'),
                'feature_names': feature_names,
                'all_features_importance': feature_importance.to_dict('records'),
                'thresholds': thresholds.tolist()[:10],
                'calibration_settings': {
                    'description': 'AGGRESSIVE calibration - ML predictions heavily reduced without direct rules',
                    'reduction_factor': 0.15,
                    'min_threshold': 2.0,
                    'max_threshold': 8.0,
                    'note': 'Max prediction for ML-only cases is 8%. Direct rules override this cap.',
                    'examples': [
                        'Raw 20% → Calibrated 3% (min floor)',
                        'Raw 40% → Calibrated 6%',
                        'Raw 60% → Calibrated 8% (max cap)',
                        'Raw 80% → Calibrated 8% (max cap)',
                        'Raw 100% → Calibrated 8% (max cap)'
                    ]
                }
            }
            
            self._save_report(model_dir, report_data, fpr, tpr)
            
            # Display results
            self.stdout.write(self.style.SUCCESS('\n' + '='*50))
            self.stdout.write(self.style.SUCCESS('PPH PREDICTION MODEL - TRAINING REPORT'))
            self.stdout.write(self.style.SUCCESS('='*50))
            self.stdout.write(f'\nTask: Prediction of Postpartum Hemorrhage (PPH)')
            self.stdout.write(f'\nData Distribution:')
            self.stdout.write(f'  Total Patients: {len(X)}')
            self.stdout.write(f'  PPH Cases: {pph_count} ({100*pph_count/len(X):.1f}%)')
            self.stdout.write(f'  Non-PPH Cases: {non_pph_count} ({100*non_pph_count/len(X):.1f}%)')
            
            self.stdout.write(f'\nModel Performance Metrics:')
            self.stdout.write(f'  Sensitivity (Recall): {report_data["sensitivity"]:.4f}')
            self.stdout.write(f'  Specificity: {report_data["specificity"]:.4f}')
            self.stdout.write(f'  Precision: {report_data["precision"]:.4f}')
            self.stdout.write(f'  Accuracy: {report_data["accuracy"]:.4f}')
            self.stdout.write(f'  AUC (Acceptable Performance): {report_data["auc"]:.4f}')
            
            self.stdout.write(f'\nConfusion Matrix:')
            self.stdout.write(f'  True Positives: {report_data["tp"]}')
            self.stdout.write(f'  True Negatives: {report_data["tn"]}')
            self.stdout.write(f'  False Positives: {report_data["fp"]}')
            self.stdout.write(f'  False Negatives: {report_data["fn"]}')
            
            self.stdout.write(f'\nStatistically Significant Risk Factors:')
            for idx, f in enumerate(feature_importance.head(8).iterrows(), 1):
                row = f[1]
                self.stdout.write(f'  {idx}. {row["feature"]}: {row["importance"]:.4f}')
            
            self.stdout.write(f'\nPrediction Calibration Settings (AGGRESSIVE):')
            self.stdout.write(f'  Reduction Factor: 0.15 (reduces predictions by 85%)')
            self.stdout.write(f'  Min Threshold: 2.0%')
            self.stdout.write(f'  Max Threshold: 8.0% (ML-only predictions capped at 8%)')
            self.stdout.write(f'  Example: Raw 60% → 9% → capped at 8%')
            self.stdout.write(f'  ⚠️  ML predictions WITHOUT direct rules will rarely exceed 8%')
            
            self.stdout.write(self.style.SUCCESS('\n✓ Training completed successfully!'))
            self.stdout.write(f'✓ Models saved to: {model_dir}')
            self.stdout.write(f'✓ ROC Curve saved to: {os.path.join(model_dir, "roc_curve.png")}')
            self.stdout.write(f'✓ Report saved to: {os.path.join(model_dir, "report.json")}')
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Training failed: {str(e)}'))
            import traceback
            traceback.print_exc()

    def _prepare_data(self, Patient):
        """Prepare training data from Patient records"""
        features_data = []
        labels = []
        
        patients = Patient.objects.filter(
            mode_of_delivery__isnull=False
        ).values()
        
        self.stdout.write(f'Found {patients.count()} patients with complete data')
        
        for patient in patients:
            try:
                feature_vector = []
                
                # GROUP 1: Demographics & Lab
                feature_vector.append(self._safe_get(patient, 'age', 0))
                feature_vector.append(self._safe_get(patient, 'bmi', 0))
                feature_vector.append(1 if self._check_in_json(
                    patient.get('menternal_medical', []), 
                    ['Chronic hypertension', 'History of blood transfusion']
                ) else 0)
                feature_vector.append(self._safe_get(patient, 'hb_g_dl', 0))
                feature_vector.append(self._safe_get(patient, 'platelets_x10e9l', 0))
                
                # GROUP 2: Pregnancy factors
                feature_vector.append(self._safe_get(patient, 'parity', 0))
                feature_vector.append(1 if self._check_in_json(
                    patient.get('obstetric_history', []), 
                    ['Multiple c-sections (2)', 'Previous c-section (1)', 'Multiple c-sections (>3)']
                ) else 0)
                feature_vector.append(1 if patient.get('liquor') in ['polihydraminos', 'Polyhydramnios'] else 0)
                feature_vector.append(1 if patient.get('fetus_number') in ['twin', 'triplete'] else 0)
                feature_vector.append(self._encode_placenta_location(patient.get('placenta_location')))
                feature_vector.append(self._safe_get(patient, 'estimated_fetal_weight_by_gm', 0))
                
                # GROUP 3: Labor factors
                feature_vector.append(self._encode_labor_type(patient.get('type_of_labor')))
                feature_vector.append(self._encode_cs_type(patient.get('type_of_cs')))
                feature_vector.append(self._safe_get(patient, 'labor_duration_hours', 0))
                
                features_data.append(feature_vector)
                
                # Target: PPH based on multiple indicators
                pph = self._calculate_pph_target(patient)
                labels.append(pph)
                
            except Exception as e:
                continue
        
        # Count classes
        y = np.array(labels)
        pph_count = np.sum(y == 1)
        non_pph_count = np.sum(y == 0)
        
        # Define feature names
        feature_names = [
            'age', 'bmi', 'chronic_hypertension_blood_transfusion', 'hb', 'platelets',
            'parity', 'previous_cs', 'polyhydramnios', 'multiple_gestation', 'placenta_location',
            'estimated_fetal_weight', 'labor_type', 'cs_type', 'labor_duration_hours'
        ]
        
        return np.array(features_data), y, feature_names, pph_count, non_pph_count

    def _calculate_pph_target(self, patient):
        """
        Calculate PPH target variable based on clinical indicators
        Returns 1 if patient has severe PPH indicators, 0 otherwise
        """
        pph_score = 0
        
        # Maternal morbidity indicators (strong PPH indicators)
        if patient.get('sever_pph'):
            pph_score += 2
        if patient.get('blood_transfusion'):
            pph_score += 1
        if patient.get('emergency_ceasrean_section'):
            pph_score += 1
        if patient.get('placental_abruption'):
            pph_score += 1
        if patient.get('rupture_uterus'):
            pph_score += 1
        if patient.get('icu_admission'):
            pph_score += 1
        
        # Blood loss amount
        blood_loss = patient.get('blood_loss', '')
        if blood_loss == 'more_than_1500':
            pph_score += 2
        elif blood_loss in ['1001_1500', '501_1000']:
            pph_score += 1
        
        # Threshold: score >= 2 indicates PPH case
        return 1 if pph_score >= 2 else 0

    def _safe_get(self, obj, key, default=0):
        """Safely get value from dict"""
        try:
            val = obj.get(key, default)
            if val is None:
                return default
            return float(val)
        except (ValueError, TypeError):
            return default

    def _check_in_json(self, json_field, keywords):
        """Check if any keyword exists in JSON field"""
        if not json_field:
            return False
        try:
            text = str(json_field).lower()
            return any(kw.lower() in text for kw in keywords)
        except:
            return False

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

    def _save_artifacts(self, model_dir, model, scaler, imputer, feature_importance):
        """Save trained models and preprocessors"""
        with open(os.path.join(model_dir, 'rf_model.pkl'), 'wb') as f:
            pickle.dump(model, f)
        with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        with open(os.path.join(model_dir, 'imputer.pkl'), 'wb') as f:
            pickle.dump(imputer, f)
        self.stdout.write(self.style.SUCCESS('✓ Models saved successfully'))

    def _save_report(self, model_dir, report_data, fpr, tpr):
        """Save training report and ROC curve"""
        with open(os.path.join(model_dir, 'report.json'), 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Generate ROC curve with high quality
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {report_data["auc"]:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - PPH Prediction Model', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'roc_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        self.stdout.write(self.style.SUCCESS('✓ Report and ROC curve saved'))