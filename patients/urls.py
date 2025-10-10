from django.urls import path
from . import views

urlpatterns = [
    # Authentication
    path('auth/register/', views.register, name='register'),
    path('auth/login/', views.login, name='login'),
    path('auth/profile/', views.user_profile, name='user_profile'),
    
    # Dashboard
    path('dashboard/stats/', views.dashboard_stats, name='dashboard_stats'),
    
    # Patients
    path('patients/', views.PatientListCreateView.as_view(), name='patient_list_create'),
    path('patients/<int:id>/', views.PatientDetailView.as_view(), name='patient_detail'),
    path('patients/search/', views.search_patients, name='search_patients'),
    
    # Labor & Delivery
    path('labor-delivery/', views.LaborDeliveryPatientsView.as_view(), name='labor_delivery'),
    
    # Operation Room
    path('operation-room/', views.OperationRoomPatientsView.as_view(), name='operation_room'),
    
    # Progress Notes
    path('patients/<int:patient_id>/notes/', views.ProgressNoteListCreateView.as_view(), name='progress_notes'),
    
    # Alerts
    path('alerts/', views.AlertListCreateView.as_view(), name='alert_list_create'),
    path('alerts/<int:pk>/acknowledge/', views.AlertAcknowledgeView.as_view(), name='alert_acknowledge'),

    # Predict CS
    path('predict-cs/', views.PredictCSView.as_view(), name='predict_cs'),
    path("patients/<identifier>/predict_pph/", views.PredictPPHView.as_view(), name="predict_pph"),
    path("patients/<identifier>/predict_neonatal/", views.PredictNeonatalView.as_view(), name="predict_neonatal"),

    # Analytics Endpoints
    path('analytics/general-indicators/', views.general_indicators, name='general_indicators'),
    path('analytics/nationality-map/', views.nationality_map, name='nationality_map'),
    path('analytics/cs-indications/', views.cs_indications_counts, name='cs_indications_counts'),
    path('analytics/risk-factors/', views.risk_factors, name='risk_factors'),
    path('analytics/maternal-outcomes/', views.maternal_outcomes, name='maternal_outcomes'),
    path('analytics/fetal-neonatal-outcomes/', views.fetal_neonatal_outcomes, name='fetal_neonatal_outcomes'),
    path('analytics/vbac-success-rate/', views.vbac_success_rate, name='vbac_success_rate'),
    path('analytics/primary-cs-rate/', views.primary_cs_rate, name='primary_cs_rate'),
    path('analytics/type-of-cs/', views.cs_type_counts, name='cs_type_counts'),
    path('analytics/special-conditions/', views.special_conditions, name='special_conditions'),

]
