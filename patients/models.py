from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.validators import MinValueValidator, MaxValueValidator


class User(AbstractUser):
    """Custom user model with role-based access"""
    ROLE_CHOICES = [
        ('admin', 'Admin'),
        ('doctor', 'Doctor'),
        ('patient', 'Patient'),
        ('team_leader', 'Team Leader'),
    ]
    
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='patient')
    phone_number = models.CharField(max_length=15, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.username} ({self.role})"


class Patient(models.Model):
    """Patient model matching the frontend form data"""
    CASE_TYPE_CHOICES = [
        ('labor', 'Labor'),
        ('operation', 'Operation'),
        ('checkup', 'Checkup'),
        ('emergency', 'Emergency'),
    ]
    
    STAGE_CHOICES = [
        ('early_labor', 'Early Labor'),
        ('active_labor', 'Active Labor'),
        ('transition', 'Transition'),
        ('pushing', 'Pushing'),
        ('delivery', 'Delivery'),
        ('recovery', 'Recovery'),
    ]
    
    BLOOD_GROUP_CHOICES = [
        ('A+', 'A+'),
        ('A-', 'A-'),
        ('B+', 'B+'),
        ('B-', 'B-'),
        ('AB+', 'AB+'),
        ('AB-', 'AB-'),
        ('O+', 'O+'),
        ('O-', 'O-'),
    ]
    
    # Basic Information
    name = models.CharField(max_length=100)
    file_number = models.CharField(max_length=20, unique=True)
    patient_id = models.CharField(max_length=20, unique=True)
    nationality = models.CharField(max_length=50)
    age = models.PositiveIntegerField(validators=[MinValueValidator(1), MaxValueValidator(120)])
    
    # Physical Measurements
    height = models.PositiveIntegerField(help_text="Height in cm")
    weight = models.DecimalField(max_digits=5, decimal_places=2, help_text="Weight in kg")
    bmi = models.DecimalField(max_digits=5, decimal_places=2)
    
    # Pregnancy Information
    gravidity = models.PositiveIntegerField(default=0, help_text="Number of pregnancies")
    parity = models.PositiveIntegerField(default=0, help_text="Number of births (parity)")
    abortion = models.PositiveIntegerField(default=0, help_text="Number of abortions")
    lmp = models.DateField(help_text="Last Menstrual Period")
    edd = models.DateField(help_text="Expected Due Date")
    gestational_age = models.CharField(max_length=20, help_text="e.g., '38 weeks 2 days'")
    BOOKING_CHOICES = [
        ("first_trimester", "First Trimester"),
        ("second_trimester", "Second Trimester"),
        ("third_trimester", "Third Trimester"),
        ("unbooked", "Unbooked"),
    ]
    booking = models.CharField(max_length=20, choices=BOOKING_CHOICES, default="unbooked")
    
    # Medical Information
    blood_group = models.CharField(max_length=5, choices=BLOOD_GROUP_CHOICES)
    
    # Vital Signs
    pulse = models.PositiveIntegerField(validators=[MinValueValidator(30), MaxValueValidator(200)])
    bp = models.CharField(max_length=15, help_text="Blood Pressure (e.g., 120/80)")
    temp = models.DecimalField(max_digits=4, decimal_places=2, help_text="Temperature in Celsius")
    oxygen_sat = models.PositiveIntegerField(
        validators=[MinValueValidator(50), MaxValueValidator(100)],
        help_text="Oxygen saturation percentage"
    )
    
    # Current Status
    case_type = models.CharField(max_length=20, choices=CASE_TYPE_CHOICES)
    room = models.CharField(max_length=20, blank=True, null=True)
    stage = models.CharField(max_length=20, choices=STAGE_CHOICES, blank=True, null=True)
    duration = models.DurationField(blank=True, null=True, help_text="Time in current stage")
    
    # Operation-specific fields
    surgeon = models.CharField(max_length=100, blank=True, null=True)
    surgery_type = models.CharField(max_length=100, blank=True, null=True)
    scheduled_time = models.DateTimeField(blank=True, null=True)
    
    # Medical History Fields (ArrayField for PostgreSQL)
    menternal_medical = models.JSONField(default=list, help_text="List of maternal medical conditions")
    obstetric_history = models.JSONField(default=list, help_text="List of obstetric history conditions")
    current_pregnancy_menternal = models.JSONField(default=list, help_text="List of current pregnancy maternal conditions")
    current_pregnancy_fetal = models.JSONField(default=list, help_text="List of current pregnancy fetal conditions")
    social = models.JSONField(default=list, help_text="List of social factors")
    
    # Delivery-specific fields
    time_of_admission = models.DateTimeField(blank=True, null=True, help_text="Time of admission to hospital")
    cervical_dilatation_at_admission = models.FloatField(blank=True, null=True, help_text="Cervical dilatation at admission in cm")
    time_of_cervix_fully_dilated = models.DateTimeField(blank=True, null=True, help_text="Time when cervix became fully dilated")
    time_of_delivery = models.DateTimeField(blank=True, null=True, help_text="Time of delivery")
    labor_duration_hours = models.FloatField(blank=True, null=True, help_text="Total labor duration in hours")
    fully_dilated_cervix = models.BooleanField(default=False, help_text="Whether cervix is fully dilated")
    
    # Maternal morbidity fields
    maternal_death = models.BooleanField(default=False)
    eclampsia = models.BooleanField(default=False)
    hellp_syndrome = models.BooleanField(default=False)
    acute_kidney_injury = models.BooleanField(default=False)
    sever_pph = models.BooleanField(default=False)
    prolonged_hospital_stay_more_4_days = models.BooleanField(default=False)
    blood_transfusion = models.BooleanField(default=False)
    emergency_ceasrean_section = models.BooleanField(default=False)
    placental_abruption = models.BooleanField(default=False)
    ceasrean_hysterectomy = models.BooleanField(default=False)
    rupture_uterus = models.BooleanField(default=False)
    readmission_within_30d = models.BooleanField(default=False)
    vte_postpartum = models.BooleanField(default=False)
    infection_endometritis = models.BooleanField(default=False)
    wound_complication = models.BooleanField(default=False)
    other_maternal_morbidity = models.TextField(blank=True, null=True)
    icu_admission = models.BooleanField(default=False)
    los_by_days = models.PositiveIntegerField(blank=True, null=True)
    ward_admissin_days = models.PositiveIntegerField(blank=True, null=True)
    icu_admission_days = models.PositiveIntegerField(blank=True, null=True)
    cost = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    
    # Neonatal fields
    birth_weight = models.FloatField(blank=True, null=True)
    apgar_score = models.PositiveIntegerField(blank=True, null=True)
    neonatal_death = models.BooleanField(default=False)
    preterm_birth_less_37_weeks = models.BooleanField(default=False)
    congenital_anomalies = models.BooleanField(default=False)
    hie = models.BooleanField(default=False)
    nicu_admission = models.BooleanField(default=False)
    birth_injuries = models.BooleanField(default=False)
    
    GENDER_CHOICES = [
        ('male', 'Male'),
        ('female', 'Female'),
        ('unknown', 'Unknown'),
    ]
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES, blank=True, null=True)
    
    PLACENTA_LOCATION_CHOICES = [
        ('upper', 'Upper'),
        ('lower', 'Lower'),
        ('covering_cervix', 'Covering Cervix'),
    ]
    placenta_location = models.CharField(max_length=20, choices=PLACENTA_LOCATION_CHOICES, blank=True, null=True)
    
    LIQUOR_CHOICES = [
        ('normal', 'Normal'),
        ('oligohydraminos', 'Oligohydramnios'),
        ('polihydraminos', 'Polihydramnios'),
    ]
    liquor = models.CharField(max_length=20, choices=LIQUOR_CHOICES, blank=True, null=True)
    estimated_fetal_weight_by_gm = models.FloatField(blank=True, null=True)
    
    # Delivery details
    PRESENTATION_CHOICES = [
        ('cephlic', 'Cephalic'),
        ('preech', 'Breech'),
        ('transverse', 'Transverse'),
        ('oblique', 'Oblique'),
    ]
    presentation = models.CharField(max_length=20, choices=PRESENTATION_CHOICES, blank=True, null=True)
    
    FETUS_NUMBER_CHOICES = [
        ('single', 'Single'),
        ('twin', 'Twin'),
        ('triplete', 'Triplet'),
    ]
    fetus_number = models.CharField(max_length=20, choices=FETUS_NUMBER_CHOICES, blank=True, null=True)
    
    GA_INTERVAL_CHOICES = [
        ('less_than_37_weeks', 'Less than 37 weeks'),
        ('37_to_40_weeks', '37 to 40 weeks'),
        ('more_than_40_weeks', 'More than 40 weeks'),
    ]
    ga_interval = models.CharField(max_length=25, choices=GA_INTERVAL_CHOICES, blank=True, null=True)
    
    TYPE_OF_LABOR_CHOICES = [
        ('spontenous_labor', 'Spontaneous Labor'),
        ('iol', 'IOL'),
        ('pre_labour_cesarean', 'Pre-labour Cesarean'),
        ('no_labour_pain', 'No Labour Pain'),
    ]
    type_of_labor = models.CharField(max_length=25, choices=TYPE_OF_LABOR_CHOICES, blank=True, null=True)
    
    MODE_OF_DELIVERY_CHOICES = [
        ('nvd', 'NVD'),
        ('cs', 'C.S'),
    ]
    mode_of_delivery = models.CharField(max_length=10, choices=MODE_OF_DELIVERY_CHOICES, blank=True, null=True)
    
    PERINEUM_INTEGRITY_CHOICES = [
        ('intact', 'Intact'),
        ('episotomy', 'Episotomy'),
        ('1st_2nd_tear', '1st/2nd Tear'),
        ('3rd_4th_tear', '3rd/4th Tear'),
        ('episotomy_3rd_4th', 'Episotomy+3rd,4th'),
    ]
    perineum_integrity = models.CharField(max_length=25, choices=PERINEUM_INTEGRITY_CHOICES, blank=True, null=True)
    
    instrumental_delivery = models.BooleanField(default=False)
    vbac = models.BooleanField(default=False)
    
    TYPE_OF_CS_CHOICES = [
        ('emergency', 'Emergency'),
        ('elective', 'Elective'),
    ]
    type_of_cs = models.CharField(max_length=15, choices=TYPE_OF_CS_CHOICES, blank=True, null=True)
    
    CS_INDICATION_CHOICES = [
        ('fetal_distress', 'Fetal distress'),
        ('failure_to_progress', 'Failure to progress'),
        ('cord_prolapse', 'Cord prolapse'),
        ('chorioamnionitis', 'Chorioamnionitis'),
        ('other_fetal_iugr', 'Other(Fetal),IUGR'),
        ('breech_baby', 'Breech baby'),
        ('multiple_pregnancy', 'Multiple pregnancy'),
        ('maternal_request_no_medical_reason', 'Maternal request (no medical reason)'),
        ('pre_eclampsia_eclampsia_hellp', 'Pre-eclampsia/eclampsia/HELLP'),
        ('maternal_medical_disease', 'Maternal medical disease'),
        ('placenta_praevia_actively_bleeding', 'Placenta praevia, actively bleeding'),
        ('placenta_praevia_not_actively_bleeding', 'Placenta praevia, not actively bleeding'),
        ('aph_intrapartum_haemorrhage', 'APH/Intrapartum haemorrhage'),
        ('placental_abruption', 'Placental abruption'),
        ('uterine_rupture', 'Uterine rupture'),
        ('other_maternal', 'Other (maternal)'),
        ('refusal_of_tolac', 'Refusal of TOLAC'),
        ('failed_tolac', 'Failed TOLAC'),
        ('feto_pelvic_disproportion', 'Feto-Pelvic Disproportion'),
        ('abnormal_lie_presentation_other_than_breech', 'Abnormal lie/presentation other than breech'),
        ('repeated_cs', 'Repeated CS'),
        ('failed_iol', 'Failed IOL'),
        ('infertility_precious_baby', 'Infertility(precious baby)'),
        ('repeated_cs_in_labor', 'Repeated CS in labor'),
    ]
    cs_indication = models.CharField(max_length=50, choices=CS_INDICATION_CHOICES, blank=True, null=True)
    
    TOTAL_NUMBER_OF_CS_CHOICES = [
        ('0', '0'),
        ('1', '1'),
        ('2', '2'),
        ('3', '3'),
        ('4', '4'),
        ('5', '5'),
        ('6', '6'),
        ('7', '7'),
        ('8', '8'),
        ('9', '9'),
        ('10', '10'),
    ]
    total_number_of_cs = models.CharField(max_length=5, choices=TOTAL_NUMBER_OF_CS_CHOICES, blank=True, null=True)
    
    PARITY_STATUS_CHOICES = [
        ('primigravida', 'Primigravida'),
        ('multipara', 'Multipara'),
    ]
    parity_status = models.CharField(max_length=15, choices=PARITY_STATUS_CHOICES, blank=True, null=True)
    robson_classification = models.TextField(blank=True, null=True)
    
    TYPE_OF_ANASTHESIA_CHOICES = [
        ('ga', 'GA'),
        ('spinal', 'Spinal'),
        ('epidural', 'Epidural'),
    ]
    type_of_anasthesia = models.CharField(max_length=15, choices=TYPE_OF_ANASTHESIA_CHOICES, blank=True, null=True)
    
    BLOOD_LOSS_CHOICES = [
        ('less_than_500', 'Less than 500'),
        ('501_1000', '501-1000'),
        ('1001_1500', '1001-1500'),
        ('more_than_1500', 'More than 1500'),
    ]
    blood_loss = models.CharField(max_length=20, choices=BLOOD_LOSS_CHOICES, blank=True, null=True)
    
    INDICATION_OF_INDUCTION_CHOICES = [
        ('post_term_pregnancy_41_weeks', 'Post-term pregnancy ≥41 weeks'),
        ('prelabor_rupture_of_membranes_prom', 'Prelabor rupture of membranes (PROM)'),
        ('oligohydramnios', 'Oligohydramnios'),
        ('gestational_hypertension', 'Gestational hypertension'),
        ('preeclampsia_diabetes_mellitus', 'Preeclampsia Diabetes mellitus'),
        ('intrauterine_growth_restriction_iugr', 'Intrauterine growth restriction (IUGR)'),
        ('intrauterine_fetal_demise_iufd', 'Intrauterine fetal demise (IUFD)'),
        ('chorioamnionitis', 'Chorioamnionitis'),
        ('elective_social_other', 'Elective (social/other)'),
    ]
    indication_of_induction = models.CharField(max_length=50, choices=INDICATION_OF_INDUCTION_CHOICES, blank=True, null=True)
    
    INDUCTION_METHOD_CHOICES = [
        ('prostaglandin_e2_gel_pessary', 'Prostaglandin E2 (gel/pessary)'),
        ('prostaglandin_e1_misoprostol', 'Prostaglandin E1 (misoprostol)'),
        ('oxytocin_infusion', 'Oxytocin infusion'),
        ('amniotomy_arm', 'Amniotomy (ARM)'),
        ('foley_catheter', 'Foley catheter'),
        ('combined_method_mechanical_pharmacological', 'Combined method (mechanical + pharmacological)'),
    ]
    induction_method = models.CharField(max_length=50, choices=INDUCTION_METHOD_CHOICES, blank=True, null=True)
    
    CERVIX_FAVRABOLE_FOR_INDUCTION_CHOICES = [
        ('unfavorable_bishop_score_less_6', 'Unfavorable (Bishop score <6)'),
        ('favorable_bishop_score_6_or_more', 'Favorable (Bishop score ≥6)'),
    ]
    cervix_favrable_for_induction = models.CharField(max_length=40, choices=CERVIX_FAVRABOLE_FOR_INDUCTION_CHOICES, blank=True, null=True)
    
    MEMBRANE_STATUS_CHOICES = [
        ('intact', 'Intact'),
        ('ruptured_spontaneous', 'Ruptured – spontaneous'),
        ('ruptured_artificial_arm', 'Ruptured – artificial (ARM)'),
    ]
    membrane_status = models.CharField(max_length=25, choices=MEMBRANE_STATUS_CHOICES, blank=True, null=True)
    
    RUPTURE_DURATION_HOUR_CHOICES = [
        ('less_than_6_hours', '<6 hours'),
        ('6_12_hours', '6–12 hours'),
        ('12_18_hours', '12–18 hours'),
        ('18_24_hours', '18–24 hours'),
        ('more_than_24_hours_prolonged_rupture', '>24 hours (prolonged rupture)'),
        ('still_intact', 'Still intact'),
    ]
    rupture_duration_hour = models.CharField(max_length=40, choices=RUPTURE_DURATION_HOUR_CHOICES, blank=True, null=True)
    
    LIQUOR_CHOICES_2 = [
        ('clear', 'Clear'),
        ('meconium_stained_thin', 'Meconium-stained (thin)'),
        ('meconium_stained_thick', 'Meconium-stained (thick)'),
        ('blood_stained', 'Blood-stained'),
        ('absent_anhydramnios_dry_tap', 'Absent (anhydramnios/dry tap)'),
    ]
    liquor_2 = models.CharField(max_length=30, choices=LIQUOR_CHOICES_2, blank=True, null=True)
    
    CTG_CATEGORY_CHOICES = [
        ('category_i_normal_reassuring', 'Category I – Normal/reassuring'),
        ('category_ii_suspicious', 'Category II – Suspicious'),
        ('category_iii_pathological', 'Category III – Pathological'),
    ]
    ctg_category = models.CharField(max_length=35, choices=CTG_CATEGORY_CHOICES, blank=True, null=True)
    
    doctor_name = models.CharField(max_length=100, blank=True, null=True)
    hb_g_dl = models.FloatField(blank=True, null=True)
    platelets_x10e9l = models.PositiveIntegerField(blank=True, null=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='created_patients')
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.file_number})"
    
    @property
    def bp_display(self):
        return self.bp
    
    def save(self, *args, **kwargs):
        # Calculate BMI if not provided
        if self.height and self.weight and not self.bmi:
            height_m = self.height / 100
            self.bmi = round(float(self.weight) / (height_m ** 2), 2)
        if "Preterm labor < 37 weeks" in self.current_pregnancy_fetal:
            self.preterm_birth_less_37_weeks = True
        super().save(*args, **kwargs)


class ProgressNote(models.Model):
    """Progress notes for patients"""
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='progress_notes')
    note = models.TextField()
    author = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Note for {self.patient.name} by {self.author}"


class Alert(models.Model):
    """Medical alerts and notifications"""
    ALERT_TYPE_CHOICES = [
        ('Critical', 'Critical'),
        ('Warning', 'Warning'),
        ('Info', 'Info'),
    ]
    
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='alerts', null=True, blank=True)
    alert_type = models.CharField(max_length=20, choices=ALERT_TYPE_CHOICES)
    message = models.TextField()
    is_acknowledged = models.BooleanField(default=False)
    acknowledged_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.alert_type.upper()}: {self.message[:50]}..."