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

from .models import User, Patient, ProgressNote, Alert
from .serializers import (
    UserSerializer, UserRegistrationSerializer, LoginSerializer,
    PatientSerializer, ProgressNoteSerializer, AlertSerializer,
    DashboardStatsSerializer
)


class PatientFilter(django_filters.FilterSet):
    """Filter for patient list view"""
    name = django_filters.CharFilter(lookup_expr='icontains')
    case_type = django_filters.ChoiceFilter(choices=Patient.CASE_TYPE_CHOICES)
    stage = django_filters.ChoiceFilter(choices=Patient.STAGE_CHOICES)
    blood_group = django_filters.ChoiceFilter(choices=Patient.BLOOD_GROUP_CHOICES)
    room = django_filters.CharFilter(lookup_expr='icontains')
    age_min = django_filters.NumberFilter(field_name='age', lookup_expr='gte')
    age_max = django_filters.NumberFilter(field_name='age', lookup_expr='lte')
    
    class Meta:
        model = Patient
        fields = ['name', 'case_type', 'stage', 'blood_group', 'room', 'age_min', 'age_max']


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
    
    def get_queryset(self):
        return Patient.objects.filter(
            case_type='labor'
        ).order_by('-created_at')


class OperationRoomPatientsView(generics.ListAPIView):
    """Get patients in operation room"""
    serializer_class = PatientSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return Patient.objects.filter(
            case_type='operation'
        ).order_by('-created_at')


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
