from django.urls import path
from . import views, cs_prediction_views, pph_prediction_views, pph_prediction_compine_new, neonatal_prediction_views

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
    path('new/predict-cs/', cs_prediction_views.predict_patient_by_identifier, name='predict_cs_new'),
    path('new/prediction-info/', cs_prediction_views.get_cs_prediction_info, name='prediction_info'),
    path('new/predict-pph/', pph_prediction_views.predict_pph, name='predict_pph_new'),
    path('new/predict-pph-compine/', pph_prediction_compine_new.predict_pph, name='predict_pph_compine_new'),
    path('new/predict-neonatal/', neonatal_prediction_views.predict_neonatal_by_identifier, name='predict_neonatal_new'),
    
    
    # Analytics Endpoints
    path('analytics/general-indicators/', views.general_indicators, name='general_indicators'),
    path('analytics/booking-status/', views.booking_status, name='booking_status'),
    path('analytics/nationality-map/', views.nationality_map, name='nationality_map'),
    path('analytics/cs-indications/', views.cs_indications_counts, name='cs_indications_counts'),
    path('analytics/risk-factors/', views.risk_factors, name='risk_factors'),
    path('analytics/maternal-outcomes/', views.maternal_outcomes, name='maternal_outcomes'),
    path('analytics/fetal-neonatal-outcomes/', views.fetal_neonatal_outcomes, name='fetal_neonatal_outcomes'),
    path('analytics/vbac-success-rate/', views.vbac_success_rate, name='vbac_success_rate'),
    path('analytics/primary-cs-rate/', views.primary_cs_rate, name='primary_cs_rate'),
    path('analytics/type-of-cs/', views.cs_type_counts, name='cs_type_counts'),
    path('analytics/special-conditions/', views.special_conditions, name='special_conditions'),
    path('analytics/mode-of-delivery-trends/', views.mode_of_delivery_trends, name='mode_of_delivery_trends'),
    path('analytics/vbac-comparison/', views.vbac_comparison, name='vbac_comparison'),
    path('analytics/primary-cs-comparison/', views.primary_cs_comparison, name='primary_cs_comparison'),
    path('analytics/instrumental-delivery-trends/', views.instrumental_delivery_trends, name='instrumental_delivery_trends'),

]
