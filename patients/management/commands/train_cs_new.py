"""
File: management/commands/train_cs_prediction.py
Django management command to train CS prediction model
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from django.core.management.base import BaseCommand
from django.db.models import Count
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, 
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
from patients.models import Patient  # Update with your app name


class Command(BaseCommand):
    help = 'Train Cesarean Section prediction model'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting CS Prediction Model Training...'))
        
        try:
            # Extract training data
            data = self.extract_training_data()
            if data.empty:
                self.stdout.write(self.style.ERROR('No training data available'))
                return
            
            self.stdout.write(f'Training samples: {len(data)}')
            
            # Train model
            model, scaler, feature_names = self.train_model(data)
            
            # Generate report
            self.generate_report(data, model, scaler, feature_names)
            
            # Save model
            self.save_model(model, scaler, feature_names)
            
            self.stdout.write(self.style.SUCCESS('Training completed successfully!'))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Training failed: {str(e)}'))

    def extract_training_data(self):
        """Extract features and labels from Patient records"""
        patients = Patient.objects.exclude(mode_of_delivery__isnull=True)
        
        data_list = []
        
        for patient in patients:
            try:
                features = self.extract_features(patient)
                label = 1 if patient.mode_of_delivery == 'cs' else 0
                features['cs'] = label
                data_list.append(features)
            except:
                continue
        
        return pd.DataFrame(data_list).dropna()

    def extract_features(self, patient):
        """Extract relevant features from patient record"""
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
        features['chronic_hypertension'] = self.check_in_json(
            patient.menternal_medical, ['chronic hypertension', 'hypertension']
        )
        features['diabetes'] = self.check_in_json(
            patient.menternal_medical, ['diabetes', 'gdm', 'gestational diabetes']
        )
        features['preeclampsia_history'] = self.check_in_json(
            patient.obstetric_history, ['preeclampsia', 'pre-eclampsia']
        )
        features['cs_history_1'] = self.check_in_json(
            patient.obstetric_history, ['previous c-section', '1 cs', 'one cs']
        )
        features['cs_history_2'] = self.check_in_json(
            patient.obstetric_history, ['2 cs', 'two cs']
        )
        features['cs_history_3plus'] = self.check_in_json(
            patient.obstetric_history, ['3 cs', '3+ cs', 'multiple cs', '4 cs', '5 cs']
        )
        features['uterine_rupture_history'] = self.check_in_json(
            patient.obstetric_history, ['uterine rupture', 'rupture']
        )
        features['cardiac_disease'] = self.check_in_json(
            patient.menternal_medical, ['cardiac', 'heart disease']
        )
        features['hiv_immunocompromised'] = self.check_in_json(
            patient.menternal_medical, ['hiv', 'immunocompromised']
        )
        features['grand_multipara'] = self.check_in_json(
            patient.social, ['grand multipara', '>=5', '>5']
        )
        
        # Current pregnancy factors
        features['multiple_gestation'] = self.check_in_json(
            patient.current_pregnancy_menternal, ['multiple gestation', 'twins', 'triplet']
        )
        features['placenta_previa'] = self.check_in_json(
            patient.current_pregnancy_menternal, ['placenta previa', 'previa']
        )
        features['placental_abruption'] = self.check_in_json(
            patient.current_pregnancy_menternal, ['abruption', 'placental abruption']
        )
        features['severe_anemia'] = self.check_in_json(
            patient.current_pregnancy_menternal, ['severe anemia', 'anemia']
        )
        features['ivf_icsi'] = self.check_in_json(
            patient.current_pregnancy_menternal, ['ivf', 'icsi']
        )
        
        # Fetal factors
        features['non_cephalic_presentation'] = self.check_in_json(
            patient.current_pregnancy_fetal, ['breech', 'transverse', 'oblique', 'non-cephalic']
        )
        features['iugr'] = self.check_in_json(
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

    def check_in_json(self, json_list, keywords):
        """Check if any keyword exists in JSON list (case-insensitive)"""
        if not json_list:
            return 0
        
        json_str = str(json_list).lower()
        for keyword in keywords:
            if keyword.lower() in json_str:
                return 1
        return 0

    def train_model(self, data):
        """Train Random Forest model"""
        X = data.drop('cs', axis=1)
        y = data['cs']
        
        feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train_scaled, y_train)
        
        # Store test data for evaluation
        self.test_data = {
            'X_test': X_test_scaled,
            'y_test': y_test,
            'X_test_unscaled': X_test,
            'feature_names': feature_names
        }
        
        return model, scaler, feature_names

    def generate_report(self, data, model, scaler, feature_names):
        """Generate performance report and visualization"""
        X_test = self.test_data['X_test']
        y_test = self.test_data['y_test']
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        sensitivity = recall_score(y_test, y_pred)  # True positive rate
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        precision = precision_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Print report
        self.stdout.write("\n" + "="*60)
        self.stdout.write("CESAREAN SECTION PREDICTION MODEL REPORT")
        self.stdout.write("="*60)
        self.stdout.write(f"\nAI TASK: Predict likelihood of CS vs SVD before labor\n")
        
        self.stdout.write(f"Training samples: {len(data)}")
        self.stdout.write(f"CS rate in dataset: {(data['cs'].sum()/len(data)*100):.1f}%\n")
        
        self.stdout.write("MODEL PERFORMANCE METRICS:")
        self.stdout.write(f"  • AUC-ROC: {auc_score:.3f}")
        self.stdout.write(f"  • Accuracy: {accuracy:.3f}")
        self.stdout.write(f"  • Sensitivity (True Positive Rate): {sensitivity:.3f}")
        self.stdout.write(f"  • Specificity (True Negative Rate): {specificity:.3f}")
        self.stdout.write(f"  • Precision: {precision:.3f}\n")
        
        self.stdout.write("TOP 15 STATISTICALLY SIGNIFICANT RISK FACTORS:")
        for idx, row in feature_importance.head(15).iterrows():
            self.stdout.write(f"  • {row['feature']}: {row['importance']:.4f}")
        
        self.stdout.write("\nCLASSIFICATION REPORT:")
        self.stdout.write(classification_report(y_test, y_pred, 
                                               target_names=['SVD', 'CS']))
        
        # Generate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - CS Prediction Model', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        
        # Save plot
        plot_path = 'cs_prediction_roc_curve.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.stdout.write(f"\n✓ ROC curve saved: {plot_path}")
        plt.close()

    def save_model(self, model, scaler, feature_names):
        """Save trained model and scaler"""
        model_dir = 'ml_models'
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'cs_prediction_model.pkl')
        scaler_path = os.path.join(model_dir, 'cs_prediction_scaler.pkl')
        features_path = os.path.join(model_dir, 'cs_prediction_features.json')
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(features_path, 'w') as f:
            json.dump(feature_names, f)
        
        self.stdout.write(f"\n✓ Model saved: {model_path}")
        self.stdout.write(f"✓ Scaler saved: {scaler_path}")
        self.stdout.write(f"✓ Features saved: {features_path}")
