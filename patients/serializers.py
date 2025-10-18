from rest_framework import serializers
from django.contrib.auth import authenticate
from .models import User, Patient, ProgressNote, Alert, Baby


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'role', 'phone_number', 'date_joined']
        read_only_fields = ['id', 'date_joined']


class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)
    password_confirm = serializers.CharField(write_only=True)
    
    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'password_confirm', 'first_name', 'last_name', 'role', 'phone_number']
    
    def validate(self, attrs):
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError("Passwords don't match")
        return attrs
    
    def create(self, validated_data):
        validated_data.pop('password_confirm')
        user = User.objects.create_user(**validated_data)
        return user


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField()
    
    def validate(self, attrs):
        email = attrs.get('email')
        password = attrs.get('password')
        
        if email and password:
            user = authenticate(username=email, password=password)
            if not user:
                raise serializers.ValidationError('Invalid credentials')
            if not user.is_active:
                raise serializers.ValidationError('User account is disabled')
            attrs['user'] = user
            return attrs
        else:
            raise serializers.ValidationError('Must include email and password')


class BabySerializer(serializers.ModelSerializer):
    class Meta:
        model = Baby
        fields = [
            'id', 'file_number', 'new_born_viability', 'preterm_birth_less_37_weeks', 'gender',
            'birth_weight', 'apgar_score', 'neonatal_death', 'congenital_anomalies', 'hie',
            'nicu_admission', 'birth_injuries'
        ]
        read_only_fields = ['id']


class PatientSerializer(serializers.ModelSerializer):
    created_by_name = serializers.CharField(source='created_by.get_full_name', read_only=True)
    babies = BabySerializer(many=True, read_only=False)
    
    class Meta:
        model = Patient
        fields = [
            'id', 'name', 'file_number', 'patient_id', 'nationality', 'age',
            'height', 'weight', 'bmi', 'gravidity', 'parity', 'abortion', 'lmp', 'edd',
            'gestational_age', 'booking', 'blood_group', 'pulse', 'bp', 'temp', 'oxygen_sat',
            'case_type', 'room', 'stage', 'duration', 'surgeon', 'surgery_type', 'scheduled_time',
            'menternal_medical', 'obstetric_history', 'current_pregnancy_menternal', 
            'current_pregnancy_fetal', 'social',
            
            # Delivery-specific fields
            'time_of_admission', 'cervical_dilatation_at_admission', 'time_of_cervix_fully_dilated',
            'time_of_delivery', 'labor_duration_hours', 'fully_dilated_cervix',
            
            # Maternal morbidity fields
            'maternal_death', 'eclampsia', 'hellp_syndrome', 'acute_kidney_injury', 'sever_pph',
            'prolonged_hospital_stay_more_4_days', 'blood_transfusion', 'emergency_ceasrean_section',
            'placental_abruption', 'ceasrean_hysterectomy', 'rupture_uterus', 'readmission_within_30d',
            'vte_postpartum', 'infection_endometritis', 'wound_complication', 'other_maternal_morbidity',
            'icu_admission', 'los_by_days', 'ward_admissin_days', 'icu_admission_days', 'cost',
            
            # Delivery details
            'presentation', 'fetus_number', 'ga_interval', 'type_of_labor', 'mode_of_delivery',
            'perineum_integrity', 'instrumental_delivery', 'vbac', 'type_of_cs', 'cs_indication',
            'total_number_of_cs', 'parity_status', 'robson_classification', 'type_of_anasthesia',
            'blood_loss', 'indication_of_induction', 'induction_method', 'cervix_favrable_for_induction',
            'membrane_status', 'rupture_duration_hour', 'liquor_2', 'ctg_category',
            'doctor_name', 'hb_g_dl', 'platelets_x10e9l',
            
            'created_at', 'updated_at', 'created_by_name',
            # Added related babies
            'babies',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'bmi']
    
    def validate_file_number(self, value):
        if Patient.objects.filter(file_number=value).exclude(pk=self.instance.pk if self.instance else None).exists():
            raise serializers.ValidationError("A patient with this file number already exists.")
        return value
    
    def validate_patient_id(self, value):
        if Patient.objects.filter(patient_id=value).exclude(pk=self.instance.pk if self.instance else None).exists():
            raise serializers.ValidationError("A patient with this patient ID already exists.")
        return value

    def update(self, instance, validated_data):
        babies_data = validated_data.pop('babies', None)
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        if babies_data is not None:
            current_babies = {int(b.id): b for b in instance.babies.all() if b.id is not None}
            sent_ids = set()
            for baby_data in babies_data:
                baby_id = baby_data.get('id', None)
                try:
                    baby_id_int = int(baby_id)
                except (TypeError, ValueError):
                    baby_id_int = None
                if baby_id_int and baby_id_int in current_babies:
                    baby = current_babies[baby_id_int]
                    for key, val in baby_data.items():
                        if key != 'id':
                            setattr(baby, key, val)
                    baby.save()
                    sent_ids.add(baby_id_int)
                else:
                    instance.babies.create(**{k: v for k, v in baby_data.items() if k != 'id'})
            for baby_id, baby in current_babies.items():
                if baby_id not in sent_ids:
                    baby.delete()
        return instance

    def create(self, validated_data):
        babies_data = validated_data.pop('babies', None)
        patient = super().create(validated_data)
        if babies_data:
            existing_babies = {int(b.id): b for b in patient.babies.all() if b.id is not None}
            sent_ids = set()
            for baby_data in babies_data:
                baby_id = baby_data.get('id', None)
                try:
                    baby_id_int = int(baby_id)
                except (TypeError, ValueError):
                    baby_id_int = None
                if baby_id_int and baby_id_int in existing_babies:
                    baby = existing_babies[baby_id_int]
                    for key, val in baby_data.items():
                        if key != 'id':
                            setattr(baby, key, val)
                    baby.save()
                    sent_ids.add(baby_id_int)
                else:
                    patient.babies.create(**{k: v for k, v in baby_data.items() if k != 'id'})
        return patient


class ProgressNoteSerializer(serializers.ModelSerializer):
    author_name = serializers.CharField(source='author.get_full_name', read_only=True)
    
    class Meta:
        model = ProgressNote
        fields = ['id', 'patient', 'note', 'author', 'author_name', 'created_at']
        read_only_fields = ['id', 'created_at']


class AlertSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='patient.name', read_only=True)
    acknowledged_by_name = serializers.CharField(source='acknowledged_by.get_full_name', read_only=True)
    
    class Meta:
        model = Alert
        fields = [
            'id', 'patient', 'patient_name', 'alert_type', 'message',
            'is_acknowledged', 'acknowledged_by', 'acknowledged_by_name',
            'acknowledged_at', 'created_at'
        ]
        read_only_fields = ['id', 'created_at', 'acknowledged_at']


class DashboardStatsSerializer(serializers.Serializer):
    """Serializer for dashboard statistics"""
    active_patients = serializers.IntegerField()
    todays_deliveries = serializers.IntegerField()
    scheduled_operations = serializers.IntegerField()
    critical_alerts = serializers.IntegerField()
