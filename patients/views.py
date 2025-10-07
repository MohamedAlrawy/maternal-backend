from rest_framework import generics, status, filters
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from django.db.models import Q, Count
from django.utils import timezone
from datetime import datetime, timedelta
import django_filters
from django_filters.rest_framework import DjangoFilterBackend

from .models import User, Patient, ProgressNote, Alert
from .serializers import (
    UserSerializer, UserRegistrationSerializer, LoginSerializer,
    PatientSerializer, ProgressNoteSerializer, AlertSerializer,
    DashboardStatsSerializer
)


class PatientFilter(django_filters.FilterSet):
    """Filter for patient list view"""
    name = django_filters.CharFilter(lookup_expr='icontains')
    file_number = django_filters.CharFilter(lookup_expr='icontains')
    case_type = django_filters.ChoiceFilter(choices=Patient.CASE_TYPE_CHOICES)
    stage = django_filters.ChoiceFilter(choices=Patient.STAGE_CHOICES)
    blood_group = django_filters.ChoiceFilter(choices=Patient.BLOOD_GROUP_CHOICES)
    room = django_filters.CharFilter(lookup_expr='icontains')
    age_min = django_filters.NumberFilter(field_name='age', lookup_expr='gte')
    age_max = django_filters.NumberFilter(field_name='age', lookup_expr='lte')
    
    class Meta:
        model = Patient
        fields = ['name', 'file_number', 'case_type', 'stage', 'blood_group', 'room', 'age_min', 'age_max']


class PatientListCreateView(generics.ListCreateAPIView):
    """List and create patients with filtering, search, and pagination"""
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [django_filters.rest_framework.DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_class = PatientFilter
    search_fields = ['name', 'file_number', 'patient_id', 'nationality']
    ordering_fields = ['name', 'age', 'created_at', 'updated_at']
    ordering = ['-created_at']
    
    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)


class PatientDetailView(generics.RetrieveUpdateDestroyAPIView):
    """Retrieve, update, or delete a patient"""
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'id'


class LaborDeliveryPatientsView(generics.ListAPIView):
    """Get patients in labor and delivery"""
    serializer_class = PatientSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['name', 'file_number', 'patient_id', 'nationality']
    ordering_fields = ['created_at']
    ordering = ['-created_at']
    filterset_class = PatientFilter

    def get_queryset(self):
        queryset = Patient.objects.filter(
            case_type='labor'
        )
        # Search and ordering handled by filter_backends
        return queryset


class OperationRoomPatientsView(generics.ListAPIView):
    """Get patients in operation room"""
    serializer_class = PatientSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['name', 'file_number', 'patient_id', 'nationality']
    ordering_fields = ['created_at']
    ordering = ['-created_at']
    filterset_class = PatientFilter

    def get_queryset(self):
        queryset = Patient.objects.filter(
            case_type='operation'
        )
        # Search and ordering handled by filter_backends
        return queryset


class ProgressNoteListCreateView(generics.ListCreateAPIView):
    """List and create progress notes"""
    serializer_class = ProgressNoteSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        patient_id = self.kwargs.get('patient_id')
        return ProgressNote.objects.filter(patient_id=patient_id).order_by('-created_at')
    
    def perform_create(self, serializer):
        patient_id = self.kwargs.get('patient_id')
        serializer.save(patient_id=patient_id, author=self.request.user)


class AlertListCreateView(generics.ListCreateAPIView):
    """List and create alerts"""
    queryset = Alert.objects.all()
    serializer_class = AlertSerializer
    permission_classes = [IsAuthenticated]
    ordering = ['-created_at']


class AlertAcknowledgeView(generics.UpdateAPIView):
    """Acknowledge an alert"""
    queryset = Alert.objects.all()
    serializer_class = AlertSerializer
    permission_classes = [IsAuthenticated]
    
    def update(self, request, *args, **kwargs):
        alert = self.get_object()
        alert.is_acknowledged = True
        alert.acknowledged_by = request.user
        alert.acknowledged_at = timezone.now()
        alert.save()
        serializer = self.get_serializer(alert)
        return Response(serializer.data)


@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    """User registration endpoint"""
    serializer = UserRegistrationSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        token, created = Token.objects.get_or_create(user=user)
        return Response({
            'user': UserSerializer(user).data,
            'token': token.key
        }, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    """User login endpoint"""
    serializer = LoginSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.validated_data['user']
        token, created = Token.objects.get_or_create(user=user)
        return Response({
            'user': UserSerializer(user).data,
            'token': token.key
        }, status=status.HTTP_200_OK)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def dashboard_stats(request):
    """Get dashboard statistics"""
    today = timezone.now().date()
    
    stats = {
        'active_patients': Patient.objects.filter(
            case_type__in=['labor', 'operation']
        ).count(),
        'todays_deliveries': Patient.objects.filter(
            case_type='labor',
            created_at__date=today
        ).count(),
        'scheduled_operations': Patient.objects.filter(
            case_type='operation'
        ).count(),
        'critical_alerts': Alert.objects.filter(
            alert_type='Critical',
            is_acknowledged=False
        ).count()
    }
    
    serializer = DashboardStatsSerializer(stats)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_profile(request):
    """Get current user profile"""
    serializer = UserSerializer(request.user)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def search_patients(request):
    """Search patients with advanced filtering"""
    query = request.GET.get('q', '')
    case_type = request.GET.get('case_type', '')
    stage = request.GET.get('stage', '')
    
    queryset = Patient.objects.all()
    
    if query:
        queryset = queryset.filter(
            Q(name__icontains=query) |
            Q(file_number__icontains=query) |
            Q(patient_id__icontains=query) |
            Q(nationality__icontains=query)
        )
    
    if case_type:
        queryset = queryset.filter(case_type=case_type)
    
    if stage:
        queryset = queryset.filter(stage=stage)
    
    serializer = PatientSerializer(queryset[:50], many=True)  # Limit to 50 results
    return Response(serializer.data)

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .predictor import predict_patient_payload, predict_patient_by_identifier

class PredictCSView(APIView):
    """
    POST patient JSON OR { "patient_id": "..."} OR {"file_number": "..."} OR {"pk": ...}
    returns prediction.
    """
    permission_classes = [AllowAny]
    def post(self, request, format=None):
        data = request.data

        # 1) If user sent DB identifier, use predictor that loads Patient from DB
        for key in ('pk', 'patient_id', 'file_number', 'id'):
            if key in data and data[key] not in (None, ''):
                identifier = data[key]
                result = predict_patient_by_identifier(identifier)
                if 'error' in result:
                    status_code = result.get('status', 400)
                    return Response(result, status=status_code)
                return Response(result, status=status.HTTP_200_OK)

        # 2) Otherwise assume full patient payload present -> call predict_patient_payload
        try:
            result = predict_patient_payload(data)
            if 'error' in result:
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': f'Prediction failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .pph_predictor import predict_pph_by_identifier


class PredictPPHView(APIView):
    """
    API endpoint to predict Postpartum Hemorrhage (PPH) risk for a patient.
    URL: /patients/<identifier>/predict_pph/
    Optional query param: ?threshold=0.5
    """
    permission_classes = [AllowAny]

    def get(self, request, identifier):
        try:
            threshold = float(request.GET.get("threshold", 0.5))
        except Exception:
            threshold = 0.5

        result = predict_pph_by_identifier(identifier, threshold=threshold)

        if isinstance(result, dict) and result.get("status") == 404:
            return Response(result, status=status.HTTP_404_NOT_FOUND)

        return Response(result, status=status.HTTP_200_OK)


from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .neonatal_predictor import predict_neonatal_by_identifier

class PredictNeonatalView(APIView):
    permission_classes = [AllowAny]
    def get(self, request, identifier):
        try:
            threshold = float(request.GET.get("threshold", 0.5))
        except Exception:
            threshold = 0.5
        try:
            rule_threshold = float(request.GET.get("rule_threshold", 80))
        except Exception:
            rule_threshold = 80
        result = predict_neonatal_by_identifier(identifier, threshold=threshold, rule_threshold=rule_threshold)
        if isinstance(result, dict) and result.get("status") == 404:
            return Response(result, status=status.HTTP_404_NOT_FOUND)
        if isinstance(result, dict) and result.get("status") == 400:
            return Response(result, status=status.HTTP_400_BAD_REQUEST)
        return Response(result, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def general_indicators(request):
    """General indicators for analytics dashboard"""
    total_patients = Patient.objects.count()
    saudi_nationals = Patient.objects.filter(nationality__iexact="saudi arabia").count()
    non_saudi_nationals = Patient.objects.exclude(nationality__iexact="saudi arabia").count()
    advanced_maternal = Patient.objects.filter(age__gte=35).count()
    obesity_prevalence = Patient.objects.filter(bmi__gte=30).count()
    primigravida = Patient.objects.filter(parity=0).count()
    multipara = Patient.objects.filter(parity__gte=1).count()
    booked_cases = Patient.objects.exclude(booking="unbooked").count()

    def percent(part, total):
        return round((part / total) * 100, 2) if total else 0

    data = {
        "total_deliveries": total_patients,
        "saudi_nationals": {"count": saudi_nationals, "percent": percent(saudi_nationals, total_patients)},
        "non_saudi_nationals": {"count": non_saudi_nationals, "percent": percent(non_saudi_nationals, total_patients)},
        "advanced_maternal": {"count": advanced_maternal, "percent": percent(advanced_maternal, total_patients)},
        "obesity_prevalence": {"count": obesity_prevalence, "percent": percent(obesity_prevalence, total_patients)},
        "primigravida": {"count": primigravida, "percent": percent(primigravida, total_patients)},
        "multipara": {"count": multipara, "percent": percent(multipara, total_patients)},
        "booked_cases": {"count": booked_cases, "percent": percent(booked_cases, total_patients)},
    }
    return Response(data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def nationality_map(request):
    """Return grouped nationality counts for map visualization"""
    # Get top 5 nationalities (case-insensitive group by)
    qs = (
        Patient.objects.values('nationality')
        .annotate(count=Count('id'))
        .order_by('-count')[:5]
    )
    # Static mapping for ISO_A3 and coordinates (extend as needed)
    country_map = {
        'saudi arabia': { 'code': 'SAU', 'coords': [45.0792, 23.8859] },
        'egypt': { 'code': 'EGY', 'coords': [30.8025, 26.8206] },
        'sudan': { 'code': 'SDN', 'coords': [30.2176, 15.5007] },
        'yemen': { 'code': 'YEM', 'coords': [48.5164, 15.5527] },
        'india': { 'code': 'IND', 'coords': [78.9629, 20.5937] },
        'pakistan': { 'code': 'PAK', 'coords': [69.3451, 30.3753] },
        # Add more as needed
    }
    result = []
    for row in qs:
        nat = (row['nationality'] or '').strip().lower()
        info = country_map.get(nat, { 'code': 'OTH', 'coords': [0,0] })
        result.append({
            'code': info['code'],
            'label': row['nationality'],
            'count': row['count'],
            'coords': info['coords'],
        })
    return Response(result)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def cs_indications_counts(request):
    """Return counts for CS indications for analytics bar chart"""
    from .models import Patient
    from django.db.models import Q
    count = Patient.objects.all().count()
    result = [
        {
            "label": "Non‑progress of labor",
            "count": Patient.objects.filter(cs_indication__icontains="failure_to_progress").count(),
            "percentage": round((Patient.objects.filter(cs_indication__icontains="failure_to_progress").count() / count) * 100, 2) if count else 0
        },
        {
            "label": "Fetal distress",
            "count": Patient.objects.filter(cs_indication__icontains="fetal_distress").count(),
            "percentage": round((Patient.objects.filter(cs_indication__icontains="fetal_distress").count() / count) * 100, 2) if count else 0
        },
        {
            "label": "Transverse/unstable lie",
            "count": Patient.objects.filter(
                Q(presentation__icontains="transverse") |
                Q(presentation__icontains="abnormal_lie_presentation_other_than_breech")
            ).count(),
            "percentage": round((Patient.objects.filter(cs_indication__icontains="fetal_distress").count() / count) * 100, 2) if count else 0
        },
        {
            "label": "Previous CS",
            "count": Patient.objects.filter(
                Q(cs_indication__icontains="repeated_cs_in_labor") |
                Q(cs_indication__icontains="repeated_cs")
            ).count(),
            "percentage": round((Patient.objects.filter(cs_indication__icontains="fetal_distress").count() / count) * 100, 2) if count else 0
        },
        {
            "label": "Placenta previa/bleeding",
            "count": Patient.objects.filter(cs_indication__icontains="placenta_praevia_actively_bleeding").count(),
            "percentage": round((Patient.objects.filter(cs_indication__icontains="fetal_distress").count() / count) * 100, 2) if count else 0
        },
        {
            "label": "Placental abruption",
            "count": Patient.objects.filter(cs_indication__icontains="placental_abruption").count(),
            "percentage": round((Patient.objects.filter(cs_indication__icontains="fetal_distress").count() / count) * 100, 2) if count else 0
        },
        {
            "label": "Breech presentation",
            "count": Patient.objects.filter(cs_indication__icontains="abnormal_lie_presentation_other_than_breech").count(),
            "percentage": round((Patient.objects.filter(cs_indication__icontains="fetal_distress").count() / count) * 100, 2) if count else 0
        },
    ]
    return Response(result, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def risk_factors(request):
    """Return counts for maternal medical risk factors"""
    from .models import Patient
    from django.db.models import Q
    
    total_patients = Patient.objects.count()
    
    def count_with_condition(condition):
        return Patient.objects.filter(menternal_medical__contains=[condition]).count()
    
    def percent(part, total):
        return round((part / total) * 100, 2) if total else 0
    
    risk_factors_data = [
        {
            "label": "Chronic hypertension",
            "count": count_with_condition("Chronic hypertension"),
            "percentage": percent(count_with_condition("Chronic hypertension"), total_patients)
        },
        {
            "label": "Diabetes",
            "count": count_with_condition("Diabetes"),
            "percentage": percent(count_with_condition("Diabetes"), total_patients)
        },
        {
            "label": "Epilepsy",
            "count": count_with_condition("Epilepsy"),
            "percentage": percent(count_with_condition("Epilepsy"), total_patients)
        },
        {
            "label": "Cardiac disease",
            "count": count_with_condition("Cardiac disease"),
            "percentage": percent(count_with_condition("Cardiac disease"), total_patients)
        },
        {
            "label": "Renal Disease",
            "count": count_with_condition("Renal Disease"),
            "percentage": percent(count_with_condition("Renal Disease"), total_patients)
        },
        {
            "label": "History of anemia",
            "count": count_with_condition("History of anemia"),
            "percentage": percent(count_with_condition("History of anemia"), total_patients)
        },
        {
            "label": "Autoimmune disease",
            "count": count_with_condition("Autoimmune disease"),
            "percentage": percent(count_with_condition("Autoimmune disease"), total_patients)
        },
        {
            "label": "Thyroid disorder",
            "count": count_with_condition("Thyroid disorder"),
            "percentage": percent(count_with_condition("Thyroid disorder"), total_patients)
        },
        {
            "label": "Thromboembolic event",
            "count": count_with_condition("Thromboembolic event"),
            "percentage": percent(count_with_condition("Thromboembolic event"), total_patients)
        },
    ]
    
    result = {
        "risk_factors": risk_factors_data,
        "total_patients": total_patients
    }
    
    return Response(result, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def maternal_outcomes(request):
    """Return counts for maternal outcomes"""
    from .models import Patient
    from django.db.models import Q
    
    total_patients = Patient.objects.count()
    
    def percent(part, total):
        return round((part / total) * 100, 2) if total else 0
    
    # PPH Rate (>1000 mL)
    pph_count = Patient.objects.filter(
        Q(blood_loss="1001_1500") | Q(blood_loss="more_than_1500")
    ).count()
    
    # Blood Transfusion Rate
    blood_transfusion_count = Patient.objects.filter(blood_transfusion=True).count()
    
    # Maternal ICU Admission Rate
    icu_admission_count = Patient.objects.filter(icu_admission=True).count()
    
    # Cesarean Hysterectomy Rate
    cesarean_hysterectomy_count = Patient.objects.filter(ceasrean_hysterectomy=True).count()
    
    # Eclampsia
    eclampsia_count = Patient.objects.filter(eclampsia=True).count()
    
    # HELLP Syndrome
    hellp_count = Patient.objects.filter(hellp_syndrome=True).count()
    
    # Acute Kidney Injury
    aki_count = Patient.objects.filter(acute_kidney_injury=True).count()
    
    # Severe PPH
    severe_pph_count = Patient.objects.filter(sever_pph=True).count()
    
    # Prolonged Hospital Stay
    prolonged_stay_count = Patient.objects.filter(prolonged_hospital_stay_more_4_days=True).count()
    
    # Uterine Rupture
    uterine_rupture_count = Patient.objects.filter(cs_indication__icontains="uterine_rupture").count()
    
    # Readmission within 30d
    readmission_count = Patient.objects.filter(readmission_within_30d=True).count()
    
    # VTE Postpartum
    vte_count = Patient.objects.filter(vte_postpartum=True).count()
    
    # Infection/Endometritis
    infection_count = Patient.objects.filter(infection_endometritis=True).count()
    
    # Wound Complication
    wound_complication_count = Patient.objects.filter(wound_complication=True).count()
    
    maternal_outcomes_data = [
        {
            "label": "PPH Rate (>1000 mL)",
            "count": pph_count,
            "percentage": percent(pph_count, total_patients)
        },
        {
            "label": "Blood Transfusion Rate",
            "count": blood_transfusion_count,
            "percentage": percent(blood_transfusion_count, total_patients)
        },
        {
            "label": "Maternal ICU Admission Rate",
            "count": icu_admission_count,
            "percentage": percent(icu_admission_count, total_patients)
        },
        {
            "label": "Cesarean Hysterectomy Rate",
            "count": cesarean_hysterectomy_count,
            "percentage": percent(cesarean_hysterectomy_count, total_patients)
        },
        {
            "label": "Eclampsia",
            "count": eclampsia_count,
            "percentage": percent(eclampsia_count, total_patients)
        },
        {
            "label": "HELLP Syndrome",
            "count": hellp_count,
            "percentage": percent(hellp_count, total_patients)
        },
        {
            "label": "Acute Kidney Injury",
            "count": aki_count,
            "percentage": percent(aki_count, total_patients)
        },
        {
            "label": "Severe PPH",
            "count": severe_pph_count,
            "percentage": percent(severe_pph_count, total_patients)
        },
        {
            "label": "Prolonged Hospital Stay",
            "count": prolonged_stay_count,
            "percentage": percent(prolonged_stay_count, total_patients)
        },
        {
            "label": "Uterine Rupture",
            "count": uterine_rupture_count,
            "percentage": percent(uterine_rupture_count, total_patients)
        },
        {
            "label": "Readmission within 30d",
            "count": readmission_count,
            "percentage": percent(readmission_count, total_patients)
        },
        {
            "label": "VTE Postpartum",
            "count": vte_count,
            "percentage": percent(vte_count, total_patients)
        },
        {
            "label": "Infection/Endometritis",
            "count": infection_count,
            "percentage": percent(infection_count, total_patients)
        },
        {
            "label": "Wound Complication",
            "count": wound_complication_count,
            "percentage": percent(wound_complication_count, total_patients)
        },
    ]
    
    result = {
        "maternal_outcomes": maternal_outcomes_data,
        "total_patients": total_patients
    }
    
    return Response(result, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def fetal_neonatal_outcomes(request):
    """Return counts for fetal & neonatal outcomes"""
    from .models import Patient
    from django.db.models import Q
    
    total_patients = Patient.objects.count()
    
    def percent(part, total):
        return round((part / total) * 100, 2) if total else 0
    
    # Stillbirth (IUFD) Rate
    stillbirth_count = Patient.objects.filter(obstetric_history__contains=["Stillbirth"]).count()
    # Neonatal Death Rate
    neonatal_death_count = Patient.objects.filter(neonatal_death=True).count()
    # Preterm Birth Rate (<37w)
    preterm_birth_count = Patient.objects.filter(current_pregnancy_fetal__contains=["Preterm labor"]).count()
    # NICU Admission Rate
    nicu_admission_count = Patient.objects.filter(nicu_admission=True).count()
    # HIE Rate
    hie_count = Patient.objects.filter(hie=True).count()
    # Congenital Anomalies Rate
    congenital_anomalies_count = Patient.objects.filter(congenital_anomalies=True).count()
    # Birth Injuries Rate
    birth_injuries_count = Patient.objects.filter(birth_injuries=True).count()
    # Low Birth Weight (<2500g)
    low_birth_weight_count = Patient.objects.filter(birth_weight__lt=2500).count()
    # Macrosomia (≥4000g)
    macrosomia_count = Patient.objects.filter(birth_weight__gte=4000).count()
    # Booked Cases Share
    booked_cases_count = Patient.objects.exclude(booking="unbooked").count()
    
    outcomes_data = [
        {
            "label": "Stillbirth (IUFD) Rate",
            "count": stillbirth_count,
            "percentage": percent(stillbirth_count, total_patients)
        },
        {
            "label": "Neonatal Death Rate",
            "count": neonatal_death_count,
            "percentage": percent(neonatal_death_count, total_patients)
        },
        {
            "label": "Preterm Birth Rate (<37w)",
            "count": preterm_birth_count,
            "percentage": percent(preterm_birth_count, total_patients)
        },
        {
            "label": "NICU Admission Rate",
            "count": nicu_admission_count,
            "percentage": percent(nicu_admission_count, total_patients)
        },
        {
            "label": "HIE Rate",
            "count": hie_count,
            "percentage": percent(hie_count, total_patients)
        },
        {
            "label": "Congenital Anomalies Rate",
            "count": congenital_anomalies_count,
            "percentage": percent(congenital_anomalies_count, total_patients)
        },
        {
            "label": "Birth Injuries Rate",
            "count": birth_injuries_count,
            "percentage": percent(birth_injuries_count, total_patients)
        },
        {
            "label": "Low Birth Weight (<2500g)",
            "count": low_birth_weight_count,
            "percentage": percent(low_birth_weight_count, total_patients)
        },
        {
            "label": "Macrosomia (≥4000g)",
            "count": macrosomia_count,
            "percentage": percent(macrosomia_count, total_patients)
        },
        {
            "label": "Booked Cases Share",
            "count": booked_cases_count,
            "percentage": percent(booked_cases_count, total_patients)
        },
    ]
    
    result = {
        "fetal_neonatal_outcomes": outcomes_data,
        "total_patients": total_patients
    }
    
    return Response(result, status=status.HTTP_200_OK)
