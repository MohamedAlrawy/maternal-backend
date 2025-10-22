"""
Django management command to train Cesarean Section prediction model
Usage: python manage.py train_cs_model
"""
from django.core.management.base import BaseCommand
from django.db.models import Q
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from patients.models import Patient


class Command(BaseCommand):
    help = 'Train Cesarean Section prediction model'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting CS prediction model training...'))
        
        # Create models directory
        model_dir = Path('ml_models/cs_prediction')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and prepare data
        df = self.load_patient_data()
        
        if df.empty or len(df) < 50:
            self.stdout.write(self.style.ERROR(
                f'Insufficient data for training. Found {len(df)} records. Need at least 50.'
            ))
            return
        
        self.stdout.write(self.style.SUCCESS(f'Loaded {len(df)} patient records'))
        
        # Feature engineering
        X, y, feature_names = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.stdout.write(self.style.SUCCESS(
            f'Training set: {len(X_train)} | Test set: {len(X_test)}'
        ))
        self.stdout.write(self.style.SUCCESS(
            f'CS cases: {y.sum()} ({y.mean()*100:.1f}%) | Non-CS: {len(y)-y.sum()}'
        ))
        
        # Train model
        model = self.train_model(X_train, y_train, X_test, y_test)
        
        # Evaluate
        metrics = self.evaluate_model(model, X_test, y_test, feature_names)
        
        # Generate reports
        self.generate_reports(model, X_test, y_test, feature_names, metrics, model_dir)
        
        # Save model
        self.save_model(model, feature_names, metrics, model_dir)
        
        self.stdout.write(self.style.SUCCESS('Training completed successfully!'))

    def load_patient_data(self):
        """Load patient data with delivery outcomes"""
        patients = Patient.objects.exclude(
            Q(mode_of_delivery__isnull=True) | Q(mode_of_delivery='')
        ).values(
            'age', 'bmi', 'menternal_medical', 'social', 'presentation',
            'fetus_number', 'cervical_dilatation_at_admission',
            'estimated_fetal_weight_by_gm', 'labor_duration_hours',
            'mode_of_delivery', 'current_pregnancy_menternal',
            'current_pregnancy_fetal', 'obstetric_history', 'ctg_category'
        )
        
        return pd.DataFrame(list(patients))

    def prepare_features(self, df):
        """Prepare features for training"""
        features = []
        feature_names = []
        
        # Age
        features.append(df['age'].fillna(df['age'].median()).values)
        feature_names.append('age')
        
        # BMI
        bmi = df['bmi'].fillna(df['bmi'].median()).values
        features.append(bmi)
        feature_names.append('bmi')
        
        # BMI categories
        features.append((bmi >= 35) & (bmi < 40))
        feature_names.append('bmi_35_to_40')
        
        features.append(bmi >= 40)
        feature_names.append('bmi_over_40')
        
        # Chronic hypertension
        features.append(df['menternal_medical'].apply(
            lambda x: self.check_condition(x, ['chronic hypertension', 'hypertension'])
        ).values)
        feature_names.append('chronic_hypertension')
        
        # Diabetes
        features.append(df['menternal_medical'].apply(
            lambda x: self.check_condition(x, ['diabetes', 'dm', 'gestational diabetes'])
        ).values)
        feature_names.append('diabetes')
        
        # Grand multipara
        features.append(df['social'].apply(
            lambda x: self.check_condition(x, ['grand multipara', 'multipara'])
        ).values)
        feature_names.append('grand_multipara')
        
        # Presentation (non-cephalic)
        features.append(df['presentation'].apply(
            lambda x: 0 if x in ['cephlic', None, ''] else 1
        ).values)
        feature_names.append('non_cephalic_presentation')
        
        # Multiple gestation
        features.append(df['fetus_number'].apply(
            lambda x: 1 if x in ['twin', 'triplete'] else 0
        ).values)
        feature_names.append('multiple_gestation')
        
        # Cervical dilatation
        features.append(df['cervical_dilatation_at_admission'].fillna(0).values)
        feature_names.append('cervical_dilatation')
        
        # Estimated fetal weight
        efw = df['estimated_fetal_weight_by_gm'].fillna(
            df['estimated_fetal_weight_by_gm'].median()
        ).values
        features.append(efw)
        feature_names.append('fetal_weight')
        
        features.append(efw >= 4000)
        feature_names.append('macrosomia')
        
        # Labor duration
        features.append(df['labor_duration_hours'].fillna(0).values)
        feature_names.append('labor_duration')
        
        # Stack features
        X = np.column_stack(features)
        
        # Target variable
        y = (df['mode_of_delivery'] == 'cs').astype(int).values
        
        return X, y, feature_names

    def check_condition(self, field, keywords):
        """Check if any keyword exists in JSON field"""
        if not field or field == '[]':
            return 0
        
        if isinstance(field, str):
            try:
                field = json.loads(field)
            except:
                field = [field]
        
        if not isinstance(field, list):
            return 0
        
        field_lower = [str(item).lower() for item in field]
        return int(any(any(kw in item for kw in keywords) for item in field_lower))

    def train_model(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        self.stdout.write('Training Random Forest model...')
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        self.stdout.write(f'Cross-validation AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})')
        
        return model

    def evaluate_model(self, model, X_test, y_test, feature_names):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        self.stdout.write(self.style.SUCCESS('\n=== Model Performance ==='))
        self.stdout.write(f'AUC: {auc:.3f}')
        self.stdout.write(f'Sensitivity (Recall): {sensitivity:.3f}')
        self.stdout.write(f'Specificity: {specificity:.3f}')
        
        self.stdout.write('\n' + classification_report(y_test, y_pred, 
                                                       target_names=['Non-CS', 'CS']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.stdout.write('\n=== Top 10 Feature Importance ===')
        for _, row in feature_importance.head(10).iterrows():
            self.stdout.write(f"{row['feature']}: {row['importance']:.4f}")
        
        return {
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'confusion_matrix': cm.tolist(),
            'feature_importance': feature_importance.to_dict('records')
        }

    def generate_reports(self, model, X_test, y_test, feature_names, metrics, model_dir):
        """Generate visualization reports"""
        # ROC Curve
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {metrics["auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('ROC Curve - Cesarean Section Prediction', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(model_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature Importance
        fi_df = pd.DataFrame(metrics['feature_importance']).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=fi_df, x='importance', y='feature', palette='viridis')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Top 10 Feature Importance - CS Prediction', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(model_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Confusion Matrix
        cm = np.array(metrics['confusion_matrix'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-CS', 'CS'],
                   yticklabels=['Non-CS', 'CS'])
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.title('Confusion Matrix - CS Prediction', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(model_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.stdout.write(self.style.SUCCESS('Reports saved to ml_models/cs_prediction/'))

    def save_model(self, model, feature_names, metrics, model_dir):
        """Save trained model and metadata"""
        # Save model
        joblib.dump(model, model_dir / 'cs_model.pkl')
        
        # Save metadata
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'feature_names': feature_names,
            'metrics': {
                'auc': float(metrics['auc']),
                'sensitivity': float(metrics['sensitivity']),
                'specificity': float(metrics['specificity'])
            },
            'feature_importance': metrics['feature_importance']
        }
        
        with open(model_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.stdout.write(self.style.SUCCESS(f'Model saved to {model_dir}'))
