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
def booking_status(request):
    """Get booking status distribution for pie chart"""
    total_patients = Patient.objects.count()
    
    # Count each booking status
    first_trimester = Patient.objects.filter(booking="first_trimester").count()
    second_trimester = Patient.objects.filter(booking="second_trimester").count()
    third_trimester = Patient.objects.filter(booking="third_trimester").count()
    unbooked = Patient.objects.filter(booking="unbooked").count()
    
    def percent(part, total):
        return round((part / total) * 100, 2) if total else 0
    
    data = {
        "total_patients": total_patients,
        "booking_data": [
            {
                "name": "First Trimester",
                "value": first_trimester,
                "percent": percent(first_trimester, total_patients)
            },
            {
                "name": "Second Trimester",
                "value": second_trimester,
                "percent": percent(second_trimester, total_patients)
            },
            {
                "name": "Third Trimester",
                "value": third_trimester,
                "percent": percent(third_trimester, total_patients)
            },
            {
                "name": "Unbooked",
                "value": unbooked,
                "percent": percent(unbooked, total_patients)
            }
        ]
    }
    
    return Response(data, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def nationality_map(request):
    """Return grouped nationality counts for map visualization with coordinates"""
    # Get all nationalities with their counts
    qs = (
        Patient.objects.values('nationality')
        .annotate(count=Count('id'))
        .order_by('-count')
    )
    # Comprehensive mapping for countries with coordinates [longitude, latitude] and ISO codes
    country_map = {
        'saudi arabia': { 'code': 'SAU', 'coords': [45.0792, 23.8859], 'name': 'Saudi Arabia' },
        'saudi': { 'code': 'SAU', 'coords': [45.0792, 23.8859], 'name': 'Saudi Arabia' },
        'egypt': { 'code': 'EGY', 'coords': [30.8025, 26.8206], 'name': 'Egypt' },
        'egyptian': { 'code': 'EGY', 'coords': [30.8025, 26.8206], 'name': 'Egypt' },
        'yemen': { 'code': 'YEM', 'coords': [48.5164, 15.5527], 'name': 'Yemen' },
        'yemeni': { 'code': 'YEM', 'coords': [48.5164, 15.5527], 'name': 'Yemen' },
        'syria': { 'code': 'SYR', 'coords': [38.9968, 34.8021], 'name': 'Syria' },
        'syrian': { 'code': 'SYR', 'coords': [38.9968, 34.8021], 'name': 'Syria' },
        'jordan': { 'code': 'JOR', 'coords': [36.2384, 30.5852], 'name': 'Jordan' },
        'jordanian': { 'code': 'JOR', 'coords': [36.2384, 30.5852], 'name': 'Jordan' },
        'palestine': { 'code': 'PSE', 'coords': [35.2332, 31.9522], 'name': 'Palestine' },
        'palestinian': { 'code': 'PSE', 'coords': [35.2332, 31.9522], 'name': 'Palestine' },
        'lebanon': { 'code': 'LBN', 'coords': [35.8623, 33.8547], 'name': 'Lebanon' },
        'lebanese': { 'code': 'LBN', 'coords': [35.8623, 33.8547], 'name': 'Lebanon' },
        'iraq': { 'code': 'IRQ', 'coords': [44.3615, 33.3128], 'name': 'Iraq' },
        'iraqi': { 'code': 'IRQ', 'coords': [44.3615, 33.3128], 'name': 'Iraq' },
        'kuwait': { 'code': 'KWT', 'coords': [47.4818, 29.3117], 'name': 'Kuwait' },
        'kuwaiti': { 'code': 'KWT', 'coords': [47.4818, 29.3117], 'name': 'Kuwait' },
        'uae': { 'code': 'ARE', 'coords': [53.8478, 23.4241], 'name': 'UAE' },
        'emirati': { 'code': 'ARE', 'coords': [53.8478, 23.4241], 'name': 'UAE' },
        'united arab emirates': { 'code': 'ARE', 'coords': [53.8478, 23.4241], 'name': 'UAE' },
        'qatar': { 'code': 'QAT', 'coords': [51.1839, 25.3548], 'name': 'Qatar' },
        'qatari': { 'code': 'QAT', 'coords': [51.1839, 25.3548], 'name': 'Qatar' },
        'bahrain': { 'code': 'BHR', 'coords': [50.5577, 26.0667], 'name': 'Bahrain' },
        'bahraini': { 'code': 'BHR', 'coords': [50.5577, 26.0667], 'name': 'Bahrain' },
        'oman': { 'code': 'OMN', 'coords': [55.9754, 21.4735], 'name': 'Oman' },
        'omani': { 'code': 'OMN', 'coords': [55.9754, 21.4735], 'name': 'Oman' },
        'sudan': { 'code': 'SDN', 'coords': [30.2176, 12.8628], 'name': 'Sudan' },
        'sudanese': { 'code': 'SDN', 'coords': [30.2176, 12.8628], 'name': 'Sudan' },
        'morocco': { 'code': 'MAR', 'coords': [-7.0926, 31.7917], 'name': 'Morocco' },
        'moroccan': { 'code': 'MAR', 'coords': [-7.0926, 31.7917], 'name': 'Morocco' },
        'algeria': { 'code': 'DZA', 'coords': [1.6596, 28.0339], 'name': 'Algeria' },
        'algerian': { 'code': 'DZA', 'coords': [1.6596, 28.0339], 'name': 'Algeria' },
        'tunisia': { 'code': 'TUN', 'coords': [9.5375, 33.8869], 'name': 'Tunisia' },
        'tunisian': { 'code': 'TUN', 'coords': [9.5375, 33.8869], 'name': 'Tunisia' },
        'libya': { 'code': 'LBY', 'coords': [17.2283, 26.3351], 'name': 'Libya' },
        'libyan': { 'code': 'LBY', 'coords': [17.2283, 26.3351], 'name': 'Libya' },
        'pakistan': { 'code': 'PAK', 'coords': [69.3451, 30.3753], 'name': 'Pakistan' },
        'pakistani': { 'code': 'PAK', 'coords': [69.3451, 30.3753], 'name': 'Pakistan' },
        'india': { 'code': 'IND', 'coords': [78.9629, 20.5937], 'name': 'India' },
        'indian': { 'code': 'IND', 'coords': [78.9629, 20.5937], 'name': 'India' },
        'bangladesh': { 'code': 'BGD', 'coords': [90.3563, 23.6850], 'name': 'Bangladesh' },
        'bangladeshi': { 'code': 'BGD', 'coords': [90.3563, 23.6850], 'name': 'Bangladesh' },
        'philippines': { 'code': 'PHL', 'coords': [121.7740, 12.8797], 'name': 'Philippines' },
        'filipino': { 'code': 'PHL', 'coords': [121.7740, 12.8797], 'name': 'Philippines' },
        'indonesia': { 'code': 'IDN', 'coords': [113.9213, -0.7893], 'name': 'Indonesia' },
        'indonesian': { 'code': 'IDN', 'coords': [113.9213, -0.7893], 'name': 'Indonesia' },
        'turkey': { 'code': 'TUR', 'coords': [35.2433, 38.9637], 'name': 'Turkey' },
        'turkish': { 'code': 'TUR', 'coords': [35.2433, 38.9637], 'name': 'Turkey' },
        'iran': { 'code': 'IRN', 'coords': [53.6880, 32.4279], 'name': 'Iran' },
        'iranian': { 'code': 'IRN', 'coords': [53.6880, 32.4279], 'name': 'Iran' },
        'afghanistan': { 'code': 'AFG', 'coords': [67.7100, 33.9391], 'name': 'Afghanistan' },
        'afghan': { 'code': 'AFG', 'coords': [67.7100, 33.9391], 'name': 'Afghanistan' },
        'somalia': { 'code': 'SOM', 'coords': [46.1996, 5.1521], 'name': 'Somalia' },
        'somali': { 'code': 'SOM', 'coords': [46.1996, 5.1521], 'name': 'Somalia' },
        'ethiopia': { 'code': 'ETH', 'coords': [40.4897, 9.1450], 'name': 'Ethiopia' },
        'ethiopian': { 'code': 'ETH', 'coords': [40.4897, 9.1450], 'name': 'Ethiopia' },
        'eritrea': { 'code': 'ERI', 'coords': [39.7823, 15.1794], 'name': 'Eritrea' },
        'eritrean': { 'code': 'ERI', 'coords': [39.7823, 15.1794], 'name': 'Eritrea' },
        'nigeria': { 'code': 'NGA', 'coords': [8.6753, 9.0820], 'name': 'Nigeria' },
        'nigerian': { 'code': 'NGA', 'coords': [8.6753, 9.0820], 'name': 'Nigeria' },
        'chad': { 'code': 'TCD', 'coords': [18.7322, 15.4542], 'name': 'Chad' },
        'chadian': { 'code': 'TCD', 'coords': [18.7322, 15.4542], 'name': 'Chad' },
        'mali': { 'code': 'MLI', 'coords': [-3.9962, 17.5707], 'name': 'Mali' },
        'malian': { 'code': 'MLI', 'coords': [-3.9962, 17.5707], 'name': 'Mali' },
        'nepal': { 'code': 'NPL', 'coords': [84.1240, 28.3949], 'name': 'Nepal' },
        'nepalese': { 'code': 'NPL', 'coords': [84.1240, 28.3949], 'name': 'Nepal' },
        'sri lanka': { 'code': 'LKA', 'coords': [80.7718, 7.8731], 'name': 'Sri Lanka' },
        'sri lankan': { 'code': 'LKA', 'coords': [80.7718, 7.8731], 'name': 'Sri Lanka' },
        'malaysia': { 'code': 'MYS', 'coords': [101.9758, 4.2105], 'name': 'Malaysia' },
        'malaysian': { 'code': 'MYS', 'coords': [101.9758, 4.2105], 'name': 'Malaysia' },
        'usa': { 'code': 'USA', 'coords': [-95.7129, 37.0902], 'name': 'USA' },
        'american': { 'code': 'USA', 'coords': [-95.7129, 37.0902], 'name': 'USA' },
        'united states': { 'code': 'USA', 'coords': [-95.7129, 37.0902], 'name': 'USA' },
        'uk': { 'code': 'GBR', 'coords': [-3.4360, 55.3781], 'name': 'UK' },
        'british': { 'code': 'GBR', 'coords': [-3.4360, 55.3781], 'name': 'UK' },
        'united kingdom': { 'code': 'GBR', 'coords': [-3.4360, 55.3781], 'name': 'UK' },
        'canada': { 'code': 'CAN', 'coords': [-106.3468, 56.1304], 'name': 'Canada' },
        'canadian': { 'code': 'CAN', 'coords': [-106.3468, 56.1304], 'name': 'Canada' },
        'australia': { 'code': 'AUS', 'coords': [133.7751, -25.2744], 'name': 'Australia' },
        'australian': { 'code': 'AUS', 'coords': [133.7751, -25.2744], 'name': 'Australia' },
        'france': { 'code': 'FRA', 'coords': [2.2137, 46.2276], 'name': 'France' },
        'french': { 'code': 'FRA', 'coords': [2.2137, 46.2276], 'name': 'France' },
        'germany': { 'code': 'DEU', 'coords': [10.4515, 51.1657], 'name': 'Germany' },
        'german': { 'code': 'DEU', 'coords': [10.4515, 51.1657], 'name': 'Germany' },
        'italy': { 'code': 'ITA', 'coords': [12.5674, 41.8719], 'name': 'Italy' },
        'italian': { 'code': 'ITA', 'coords': [12.5674, 41.8719], 'name': 'Italy' },
        'spain': { 'code': 'ESP', 'coords': [-3.7492, 40.4637], 'name': 'Spain' },
        'spanish': { 'code': 'ESP', 'coords': [-3.7492, 40.4637], 'name': 'Spain' },
        'china': { 'code': 'CHN', 'coords': [104.1954, 35.8617], 'name': 'China' },
        'chinese': { 'code': 'CHN', 'coords': [104.1954, 35.8617], 'name': 'China' },
        'japan': { 'code': 'JPN', 'coords': [138.2529, 36.2048], 'name': 'Japan' },
        'japanese': { 'code': 'JPN', 'coords': [138.2529, 36.2048], 'name': 'Japan' },
        'south korea': { 'code': 'KOR', 'coords': [127.7669, 35.9078], 'name': 'South Korea' },
        'korean': { 'code': 'KOR', 'coords': [127.7669, 35.9078], 'name': 'South Korea' },
    }
    result = []
    total_patients = 0
    unknown_count = 0
    for row in qs:
        nat = (row['nationality'] or '').strip().lower()
        count = row['count']
        total_patients += count
        info = country_map.get(nat)
        if info:
            result.append({
                'code': info['code'],
                'label': info['name'],
                'original_nationality': row['nationality'],
                'count': count,
                'coords': info['coords'],
            })
        else:
            unknown_count += count
    if unknown_count > 0:
        result.append({
            'code': 'OTH',
            'label': 'Other',
            'original_nationality': 'Unknown',
            'count': unknown_count,
            'coords': [0, 0],
        })
    return Response({'map_data': result, 'total_patients': total_patients}, status=status.HTTP_200_OK)


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


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def vbac_success_rate(request):
    """Return monthly VBAC success rates"""
    from .models import Patient
    from django.db.models import Count
    from django.db.models.functions import TruncMonth
    
    # Get VBAC data grouped by month
    vbac_data = (
        Patient.objects
        .filter(vbac=True)
        .annotate(month=TruncMonth('time_of_admission'))
        .values('month')
        .annotate(count=Count('id'))
        .order_by('month')
    )
    
    # Format data for chart
    result = []
    for item in vbac_data:
        if item['month']:
            result.append({
                'month': item['month'].strftime('%Y-%m'),
                'month_label': item['month'].strftime('%B %Y'),
                'count': item['count']
            })
    
    return Response(result, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def primary_cs_rate(request):
    """Return monthly primary CS rates (first-time cesarean sections)"""
    from .models import Patient
    from django.db.models import Count
    from django.db.models.functions import TruncMonth
    
    # Get primary CS data grouped by month
    # Primary CS = total_number_of_cs is '0' and mode_of_delivery is 'cs'
    primary_cs_data = (
        Patient.objects
        .filter(total_number_of_cs='0', mode_of_delivery='cs')
        .annotate(month=TruncMonth('time_of_admission'))
        .values('month')
        .annotate(count=Count('id'))
        .order_by('month')
    )
    
    # Format data for chart
    result = []
    for item in primary_cs_data:
        if item['month']:
            result.append({
                'month': item['month'].strftime('%Y-%m'),
                'month_label': item['month'].strftime('%B %Y'),
                'count': item['count']
            })
    
    return Response(result, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def cs_type_counts(request):
    """Return counts for Emergency vs Elective C-sections."""
    from .models import Patient
    total_cs = Patient.objects.filter(mode_of_delivery__iexact='cs').count()
    emergency_cs = Patient.objects.filter(mode_of_delivery__iexact='cs', type_of_cs__iexact='emergency').count()
    elective_cs = Patient.objects.filter(mode_of_delivery__iexact='cs', type_of_cs__iexact='elective').count()

    def percent(part, total):
        return round((part / total) * 100, 2) if total else 0

    result = [
        {
            'label': 'Emergency CS',
            'count': emergency_cs,
            'percentage': percent(emergency_cs, total_cs),
        },
        {
            'label': 'Elective CS',
            'count': elective_cs,
            'percentage': percent(elective_cs, total_cs),
        },
        {
            'label': 'All CS',
            'count': total_cs,
            'percentage': 100,
        }
    ]
    return Response(result, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def special_conditions(request):
    """Bar chart special conditions: recurrent miscarriage, multiple gestation, GDM, Pre-eclampsia, severe anemia, placenta previa/abruption, IVF/ICSI, non-cephalic, IUGR."""
    from .models import Patient
    total_patients = Patient.objects.count()
    def percent(count):
        return round((count / total_patients) * 100, 2) if total_patients else 0

    recurrent_miscarriage = Patient.objects.filter(obstetric_history__contains=["History of recurrent miscarriage"]).count()
    multiple_gestation = Patient.objects.filter(current_pregnancy_menternal__contains=["Multiple gestation"]).count()
    gdm = Patient.objects.filter(current_pregnancy_menternal__contains=["GDM"]).count()
    preeclampsia = Patient.objects.filter(current_pregnancy_menternal__contains=["Pre-eclampsia"]).count()
    severe_anemia = Patient.objects.filter(current_pregnancy_menternal__contains=["Severe anemia (Hb<7)"]).count()
    placenta_prev_abrupt = Patient.objects.filter(current_pregnancy_menternal__contains=["Placenta previa/abruption"]).count()
    post_ivf = Patient.objects.filter(current_pregnancy_menternal__contains=["Pregnancy post IVF /ICSI"]).count()
    non_cephalic = Patient.objects.filter(current_pregnancy_menternal__contains=["Non-cephalic presentation"]).count()
    iugr = Patient.objects.filter(current_pregnancy_menternal__contains=["IUGR"]).count()

    data = [
        {"label": "History of recurrent miscarriage", "count": recurrent_miscarriage, "percentage": percent(recurrent_miscarriage)},
        {"label": "Multiple gestation", "count": multiple_gestation, "percentage": percent(multiple_gestation)},
        {"label": "GDM", "count": gdm, "percentage": percent(gdm)},
        {"label": "Pre-eclampsia", "count": preeclampsia, "percentage": percent(preeclampsia)},
        {"label": "Severe anemia (Hb<7)", "count": severe_anemia, "percentage": percent(severe_anemia)},
        {"label": "Placenta previa/abruption", "count": placenta_prev_abrupt, "percentage": percent(placenta_prev_abrupt)},
        {"label": "Pregnancy post IVF /ICSI", "count": post_ivf, "percentage": percent(post_ivf)},
        {"label": "Non-cephalic presentation", "count": non_cephalic, "percentage": percent(non_cephalic)},
        {"label": "IUGR", "count": iugr, "percentage": percent(iugr)}
    ]
    return Response({"special_conditions": data, "total_patients": total_patients}, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def mode_of_delivery_trends(request):
    """
    Get mode of delivery trends over time with optional date range filtering.
    Query params:
    - start_date: YYYY-MM-DD (optional)
    - end_date: YYYY-MM-DD (optional)
    - forecast: boolean (optional, default False) - whether to include forecast
    """
    from datetime import datetime, timedelta
    from django.db.models import Count
    from django.db.models.functions import TruncMonth
    
    # Get query parameters
    start_date_str = request.GET.get('start_date')
    end_date_str = request.GET.get('end_date')
    include_forecast = request.GET.get('forecast', 'false').lower() == 'true'
    
    # Build base queryset
    queryset = Patient.objects.filter(time_of_admission__isnull=False)
    
    # Apply date filtering if provided
    if start_date_str:
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            queryset = queryset.filter(time_of_admission__gte=start_date)
        except ValueError:
            pass
    
    if end_date_str:
        try:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            queryset = queryset.filter(time_of_admission__lte=end_date)
        except ValueError:
            pass
    
    # Group by month and mode of delivery
    cs_by_month = (
        queryset.filter(mode_of_delivery='cs')
        .annotate(month=TruncMonth('time_of_admission'))
        .values('month')
        .annotate(count=Count('id'))
        .order_by('month')
    )
    
    nvd_by_month = (
        queryset.filter(mode_of_delivery='nvd')
        .annotate(month=TruncMonth('time_of_admission'))
        .values('month')
        .annotate(count=Count('id'))
        .order_by('month')
    )
    
    # Format the data for the frontend
    cs_data = {item['month'].strftime('%Y-%m'): item['count'] for item in cs_by_month}
    nvd_data = {item['month'].strftime('%Y-%m'): item['count'] for item in nvd_by_month}
    
    # Get all unique months
    all_months = sorted(set(list(cs_data.keys()) + list(nvd_data.keys())))
    
    # Build response data
    trend_data = []
    for month in all_months:
        trend_data.append({
            'month': month,
            'month_label': datetime.strptime(month, '%Y-%m').strftime('%b %Y'),
            'cs_count': cs_data.get(month, 0),
            'nvd_count': nvd_data.get(month, 0),
            'total': cs_data.get(month, 0) + nvd_data.get(month, 0)
        })
    
    # Simple forecast logic (if requested)
    forecast_data = []
    if include_forecast and len(trend_data) >= 3:
        # Calculate average growth for last 3 months
        last_3_cs = [item['cs_count'] for item in trend_data[-3:]]
        last_3_nvd = [item['nvd_count'] for item in trend_data[-3:]]
        
        avg_cs = sum(last_3_cs) / len(last_3_cs)
        avg_nvd = sum(last_3_nvd) / len(last_3_nvd)
        
        # Generate 3 months forecast
        last_month = datetime.strptime(all_months[-1], '%Y-%m')
        for i in range(1, 4):
            forecast_month = last_month + timedelta(days=30 * i)
            forecast_month_str = forecast_month.strftime('%Y-%m')
            forecast_data.append({
                'month': forecast_month_str,
                'month_label': forecast_month.strftime('%b %Y') + ' (Forecast)',
                'cs_count': int(avg_cs),
                'nvd_count': int(avg_nvd),
                'total': int(avg_cs + avg_nvd),
                'is_forecast': True
            })
    
    return Response({
        'trends': trend_data,
        'forecast': forecast_data,
        'summary': {
            'total_cs': sum(item['cs_count'] for item in trend_data),
            'total_nvd': sum(item['nvd_count'] for item in trend_data),
            'total_deliveries': sum(item['total'] for item in trend_data)
        }
    }, status=status.HTTP_200_OK)
