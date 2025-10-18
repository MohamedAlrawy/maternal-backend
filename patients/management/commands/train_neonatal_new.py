# File: your_app/management/commands/train_neonatal_model.py

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from django.core.management.base import BaseCommand
from django.apps import apps
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, 
    classification_report, auc
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class Command(BaseCommand):
    help = 'Train neonatal complication prediction model'

    def add_arguments(self, parser):
        parser.add_argument(
            '--output-dir',
            type=str,
            default='ml_models',
            help='Directory to save trained models'
        )

    def handle(self, *args, **options):
        output_dir = options['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        self.stdout.write(self.style.SUCCESS('Starting model training...'))
        
        try:
            Patient = apps.get_model('patients', 'Patient')  # Change 'your_app' to your app name
            
            # Fetch data
            patients = Patient.objects.all().values()
            if not patients:
                self.stdout.write(self.style.ERROR('No patient data found'))
                return
            
            df = pd.DataFrame(list(patients))
            self.stdout.write(f'Loaded {len(df)} patient records')
            
            # Feature engineering
            X, y, feature_names, label_encoder_dict = self._engineer_features(df)
            
            if len(X) < 10:
                self.stdout.write(self.style.ERROR('Insufficient data for training'))
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
            
            # Metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Save model
            model_path = os.path.join(output_dir, 'neonatal_model.pkl')
            scaler_path = os.path.join(output_dir, 'neonatal_scaler.pkl')
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Save encoders and feature names
            metadata = {
                'feature_names': feature_names,
                'label_encoder_dict': {k: v.classes_.tolist() for k, v in label_encoder_dict.items()},
                'trained_date': datetime.now().isoformat(),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            metadata_path = os.path.join(output_dir, 'model_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Generate report
            report_content = f"""
NEONATAL COMPLICATION PREDICTION MODEL - TRAINING REPORT
{'=' * 70}

Dataset Information:
- Total Records: {len(df)}
- Training Samples: {len(X_train)}
- Test Samples: {len(X_test)}
- Positive Cases (Complications): {sum(y)}
- Negative Cases (No Complications): {len(y) - sum(y)}

Model Performance:
- AUC-ROC: {auc_score:.4f}
- Sensitivity (Recall): {sensitivity:.4f} ({sensitivity*100:.2f}%)
- Specificity: {specificity:.4f} ({specificity*100:.2f}%)
- Precision: {report['1']['precision']:.4f}
- F1-Score: {report['1']['f1-score']:.4f}

Top 15 Most Important Features:
{feature_importance.head(15).to_string(index=False)}

Statistically Significant Risk Factors:
{self._get_significant_factors(feature_importance)}

Model Files:
- Model: {model_path}
- Scaler: {scaler_path}
- Metadata: {metadata_path}
- ROC Curve: {os.path.join(output_dir, 'roc_curve.png')}

Training completed at: {datetime.now().isoformat()}
"""
            
            # Save ROC curve
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve - Neonatal Complication Prediction', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right", fontsize=11)
            plt.grid(alpha=0.3)
            roc_path = os.path.join(output_dir, 'roc_curve.png')
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save report
            report_path = os.path.join(output_dir, 'training_report.txt')
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            self.stdout.write(self.style.SUCCESS(report_content))
            self.stdout.write(self.style.SUCCESS(f'\nAll files saved to: {output_dir}'))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error during training: {str(e)}'))
            raise

    def _engineer_features(self, df):
        """Engineer features from patient data"""
        feature_data = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            try:
                features = {}
                
                # Basic demographics
                features['age'] = float(row.get('age') or 0)
                features['bmi'] = float(row.get('bmi') or 0)
                
                # Vital signs
                features['pulse'] = float(row.get('pulse') or 0)
                features['temp'] = float(row.get('temp') or 0)
                features['oxygen_sat'] = float(row.get('oxygen_sat') or 0)
                
                # Pregnancy factors
                features['gravidity'] = float(row.get('gravidity') or 0)
                features['parity'] = float(row.get('parity') or 0)
                features['gestational_age'] = self._parse_gestational_age(row.get('gestational_age', ''))
                
                # Medical history
                menternal_medical = row.get('menternal_medical') or []
                obstetric_history = row.get('obstetric_history') or []
                current_pregnancy_maternal = row.get('current_pregnancy_menternal') or []
                current_pregnancy_fetal = row.get('current_pregnancy_fetal') or []
                social = row.get('social') or []
                
                features['chronic_hypertension'] = 1 if self._contains_keyword(menternal_medical, 'Chronic hypertension') else 0
                features['diabetes'] = 1 if self._contains_keyword(menternal_medical, 'Diabetes') else 0
                features['multiple_cs'] = 1 if self._contains_keyword(obstetric_history, 'Multiple c-sections') else 0
                features['grand_multipara'] = 1 if self._contains_keyword(social, 'Grand multipara') else 0
                
                # Current pregnancy complications
                features['placenta_abruption'] = 1 if self._contains_keyword(current_pregnancy_maternal, 'Placental abruption') else 0
                features['placenta_previa'] = 1 if self._contains_keyword(current_pregnancy_maternal, 'Placenta previa') else 0
                features['multiple_gestation'] = 1 if self._contains_keyword(current_pregnancy_maternal, 'Multiple gestation') else 0
                features['preeclampsia'] = 1 if self._contains_keyword(current_pregnancy_maternal, 'Pre-eclampsia') else 0
                features['severe_anemia'] = 1 if self._contains_keyword(current_pregnancy_maternal, 'Severe anemia') else 0
                features['ivf_pregnancy'] = 1 if self._contains_keyword(current_pregnancy_maternal, 'IVF') else 0
                features['polyhydramios'] = 1 if self._contains_keyword(row.get('liquor', ''), 'polihydraminos') else 0
                
                # Fetal factors
                features['non_cephalic'] = 1 if row.get('presentation') not in ['cephlic', None, ''] else 0
                features['iugr'] = 1 if self._contains_keyword(current_pregnancy_fetal, 'IUGR') else 0
                features['preterm'] = 1 if self._contains_keyword(current_pregnancy_fetal, 'Preterm') else 0
                features['fetal_weight'] = float(row.get('estimated_fetal_weight_by_gm') or 0)
                
                # Delivery factors
                features['ctg_abnormal'] = 1 if row.get('ctg_category') in ['category_ii_suspicious', 'category_iii_pathological'] else 0
                features['labor_duration'] = float(row.get('labor_duration_hours') or 0)
                features['prom'] = 1 if row.get('indication_of_induction') == 'prelabor_rupture_of_membranes_prom' else 0
                features['prolonged_rupture'] = 1 if row.get('rupture_duration_hour') in ['18_24_hours', 'more_than_24_hours_prolonged_rupture'] else 0
                features['placenta_location_lower'] = 1 if row.get('placenta_location') == 'lower' else 0
                
                # Mode of delivery
                features['cs_delivery'] = 1 if row.get('mode_of_delivery') == 'cs' else 0
                features['emergency_cs'] = 1 if row.get('type_of_cs') == 'emergency' else 0
                
                # Labs
                features['hb'] = float(row.get('hb_g_dl') or 7.5)
                features['platelets'] = float(row.get('platelets_x10e9l') or 150)
                
                feature_data.append(features)
                valid_indices.append(idx)
                
            except Exception:
                continue
        
        # Target variable
        y_data = []
        for idx in valid_indices:
            row = df.loc[idx]
            # Handle apgar_score safely
            apgar_score = row.get('apgar_score')
            apgar_low = False
            try:
                if apgar_score is not None and pd.notna(apgar_score):
                    apgar_low = int(float(apgar_score)) < 7
            except (ValueError, TypeError):
                apgar_low = False
            
            target = 1 if (
                apgar_low or
                row.get('birth_injuries') or
                row.get('hie') or
                row.get('nicu_admission') or
                row.get('neonatal_death') or
                row.get('preterm_birth_less_37_weeks')
            ) else 0
            y_data.append(target)
        
        features_df = pd.DataFrame(feature_data)
        feature_names = list(features_df.columns)
        X = features_df.values
        y = np.array(y_data)
        
        return X, y, feature_names, {}

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

    def _get_significant_factors(self, feature_importance, threshold=0.05):
        """Get statistically significant factors"""
        total_importance = feature_importance['importance'].sum()
        feature_importance['percentage'] = (feature_importance['importance'] / total_importance * 100)
        significant = feature_importance[feature_importance['percentage'] >= threshold]
        
        if significant.empty:
            return "No highly significant factors identified"
        
        return "\n".join([
            f"  - {row['feature']}: {row['percentage']:.2f}%"
            for _, row in significant.iterrows()
        ])
