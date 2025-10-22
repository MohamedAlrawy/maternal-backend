"""
Django management command to train neonatal complication prediction model
Usage: python manage.py train_neonatal_model
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from django.core.management.base import BaseCommand
from django.conf import settings
from django.db.models import Q
from patients.models import Patient
import warnings
warnings.filterwarnings('ignore')


class Command(BaseCommand):
    help = 'Train neonatal complication prediction model'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting neonatal model training...'))
        
        # Prepare data
        X, y, feature_names = self.prepare_training_data()
        
        if len(X) < 30:
            self.stdout.write(self.style.WARNING(
                f'Insufficient training data: {len(X)} records. Need at least 30 records.'
            ))
            return

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
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        auc = roc_auc_score(y_test, y_pred_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        self.stdout.write(self.style.SUCCESS('\n=== MODEL PERFORMANCE ==='))
        self.stdout.write(f'AUC Score: {auc:.4f}')
        self.stdout.write(f'Sensitivity (Recall): {sensitivity:.4f}')
        self.stdout.write(f'Specificity: {specificity:.4f}')
        self.stdout.write(f'\n{classification_report(y_test, y_pred)}')

        # Feature importance
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        self.stdout.write(self.style.SUCCESS('\n=== TOP RISK FACTORS ==='))
        self.stdout.write(feature_importance_df.head(10).to_string(index=False))

        # Save model
        model_dir = os.path.join(settings.BASE_DIR, 'ml_models')
        os.makedirs(model_dir, exist_ok=True)

        with open(os.path.join(model_dir, 'neonatal_model.pkl'), 'wb') as f:
            pickle.dump(model, f)
        with open(os.path.join(model_dir, 'neonatal_scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        with open(os.path.join(model_dir, 'neonatal_features.pkl'), 'wb') as f:
            pickle.dump(feature_names, f)

        # Generate ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Neonatal Complication Prediction - ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(model_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Save report
        report = {
            'task': 'Prediction of neonatal complications',
            'auc': float(auc),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'positive_cases': int(y.sum()),
            'top_risk_factors': feature_importance_df.head(10).to_dict('records')
        }

        with open(os.path.join(model_dir, 'model_report.pkl'), 'wb') as f:
            pickle.dump(report, f)

        self.stdout.write(self.style.SUCCESS('\nModel training completed and saved successfully!'))

    def prepare_training_data(self):
        """Prepare training data from database"""
        patients = Patient.objects.all()
        
        # Define target: neonatal complication (NICU, HIE, death, preterm)
        data = []
        labels = []

        for patient in patients:
            features = self.extract_features(patient)
            if features is None:
                continue

            # Target: has neonatal complication
            target = self.has_neonatal_complication(patient)
            data.append(features)
            labels.append(target)

        if not data:
            return np.array([]), np.array([]), []

        X = np.array(data)
        y = np.array(labels)
        feature_names = [
            'age', 'bmi', 'gestational_age_weeks', 'ctg_category_iii', 'ctg_category_ii',
            'placenta_abruption', 'placenta_previa', 'multiple_gestation', 'non_cephalic',
            'prolonged_rom', 'prom_induction', 'preeclampsia', 'diabetes', 'severe_anemia',
            'polyhydramnios', 'high_birth_weight', 'post_ivf', 'preterm_labor',
            'chronic_hypertension', 'multiple_cs', 'grand_multipara', 'iugr',
            'oligohydramnios', 'history_preterm_birth'
        ]

        return X, y, feature_names

    def extract_features(self, patient):
        """Extract features from patient"""
        try:
            features = []

            # Basic features
            features.append(float(patient.age) if patient.age else 0)
            features.append(float(patient.bmi) if patient.bmi else 0)

            # Gestational age in weeks
            ga_weeks = self.parse_gestational_age(patient.gestational_age)
            features.append(float(ga_weeks) if ga_weeks else 0)

            # CTG categories
            features.append(1 if patient.ctg_category == 'category_iii_pathological' else 0)
            features.append(1 if patient.ctg_category == 'category_ii_suspicious' else 0)

            # Current pregnancy maternal conditions
            current_preg_maternal = patient.current_pregnancy_menternal or []
            features.append(1 if 'Placental abruption' in current_preg_maternal else 0)
            features.append(1 if 'Placenta previa' in current_preg_maternal else 0)
            features.append(1 if 'Multiple gestation' in current_preg_maternal else 0)
            features.append(1 if any('preeclampsia' in str(x).lower() for x in current_preg_maternal) else 0)
            features.append(1 if any('polyhydramnios' in str(x).lower() for x in current_preg_maternal) else 0)
            features.append(1 if any('oligohydramnios' in str(x).lower() for x in current_preg_maternal) else 0)

            # Current pregnancy fetal conditions
            current_preg_fetal = patient.current_pregnancy_fetal or []
            features.append(1 if 'Non-cephalic presentation' in current_preg_fetal else 0)
            features.append(1 if 'IUGR' in current_preg_fetal else 0)
            features.append(1 if 'Preterm labor' in current_preg_fetal else 0)

            # ROM duration
            features.append(1 if patient.rupture_duration_hour in 
                          ['18_24_hours', 'more_than_24_hours_prolonged_rupture'] else 0)

            # Induction indication
            features.append(1 if patient.indication_of_induction == 'prelabor_rupture_of_membranes_prom' else 0)

            # Maternal medical history
            maternal_medical = patient.menternal_medical or []
            features.append(1 if 'Diabetes' in maternal_medical else 0)
            features.append(1 if 'Chronic hypertension' in maternal_medical else 0)

            # Birth weight
            features.append(1 if patient.estimated_fetal_weight_by_gm and 
                          float(patient.estimated_fetal_weight_by_gm) >= 4000 else 0)

            # Post IVF
            features.append(1 if any('IVF' in str(x) for x in current_preg_maternal) else 0)

            # Obstetric history
            obstetric_history = patient.obstetric_history or []
            features.append(1 if any('Multiple c-sections' in str(x) for x in obstetric_history) else 0)

            # Social factors
            social = patient.social or []
            features.append(1 if 'Grand multipara' in social else 0)

            # Anemia check
            features.append(1 if patient.hb_g_dl and float(patient.hb_g_dl) < 7 else 0)

            # History of preterm birth
            features.append(1 if 'History of preterm birth' in obstetric_history else 0)

            return features

        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def parse_gestational_age(self, ga_string):
        """Parse gestational age string to weeks"""
        if not ga_string:
            return None
        try:
            # Format: "38 weeks 2 days"
            weeks = int(ga_string.split()[0])
            return weeks
        except:
            return None

    def has_neonatal_complication(self, patient):
        """Determine if patient has neonatal complication"""
        complications = (
            patient.nicu_admission or
            patient.hie or
            patient.neonatal_death or
            patient.preterm_birth_less_37_weeks or
            patient.birth_injuries or
            patient.congenital_anomalies or
            (patient.apgar_score and patient.apgar_score < 7) or
            (patient.birth_weight and patient.birth_weight < 1500)
        )
        return 1 if complications else 0
