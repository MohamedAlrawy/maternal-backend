"""
Django REST Framework API views for neonatal complication prediction
Add to your urls.py:
    path('api/predict-neonatal/', predict_neonatal_complication, name='predict_neonatal'),
    path('api/model-report/', get_model_report, name='model_report'),
"""

from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from .predict_neonatal import predict_patient_by_identifier, NeonatalPredictionService
import os
import pickle
from django.conf import settings
from rest_framework.permissions import AllowAny

@api_view(['GET'])
@permission_classes([AllowAny])
def predict_neonatal_complication(request):
    """
    Predict neonatal complications for a patient
    
    Request body:
    {
        "patient_id": "161616",  # OR
        "file_number": "161616"
    }
    
    Response:
    {
        "success": true,
        "patient_id": "161616",
        "file_number": "161616",
        "patient_name": "John Doe",
        "neonatal_complication_probability": 60,
        "neonatal_impacts": ["NICU", "HIE"],
        "prediction_method": "direct_rule",
        "all_reasons": [
            "Immature lungs/organs increase need for NICU and complications",
            "Suggests hypoxia/acidosis → urgent delivery, high neonatal risk"
        ],
        "confidence": "high",
        "risk_factors": [
            "ctg_category_iii_pathological",
            "ctg_category_ii_suspicious"
        ],
        "model_details": null,
        "error": null
    }
    """
    try:
        patient_id = request.GET.get('patient_id')
        file_number = request.GET.get('file_number')

        if not patient_id and not file_number:
            return Response(
                {
                    'success': False,
                    'error': 'Either patient_id or file_number is required'
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        result = predict_patient_by_identifier(patient_id, file_number)
        
        if result['success']:
            return Response(result, status=status.HTTP_200_OK)
        else:
            return Response(result, status=status.HTTP_404_NOT_FOUND)

    except Exception as e:
        return Response(
            {
                'success': False,
                'error': f'Server error: {str(e)}'
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def get_model_report(request):
    """
    Get model performance report and statistics
    
    Response:
    {
        "success": true,
        "task": "Prediction of neonatal complications",
        "auc": 0.8234,
        "sensitivity": 0.82,
        "specificity": 0.75,
        "training_samples": 250,
        "test_samples": 50,
        "positive_cases": 145,
        "top_risk_factors": [
            {
                "feature": "ctg_category_iii",
                "importance": 0.2234
            },
            ...
        ]
    }
    """
    try:
        model_dir = os.path.join(settings.BASE_DIR, 'ml_models')
        report_path = os.path.join(model_dir, 'model_report.pkl')

        if not os.path.exists(report_path):
            return Response(
                {
                    'success': False,
                    'error': 'Model report not found. Please train the model first.'
                },
                status=status.HTTP_404_NOT_FOUND
            )

        with open(report_path, 'rb') as f:
            report = pickle.load(f)

        return Response(
            {
                'success': True,
                'task': report.get('task'),
                'auc': report.get('auc'),
                'sensitivity': report.get('sensitivity'),
                'specificity': report.get('specificity'),
                'training_samples': report.get('training_samples'),
                'test_samples': report.get('test_samples'),
                'positive_cases': report.get('positive_cases'),
                'top_risk_factors': report.get('top_risk_factors', [])
            },
            status=status.HTTP_200_OK
        )

    except Exception as e:
        return Response(
            {
                'success': False,
                'error': f'Server error: {str(e)}'
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def get_roc_curve_image(request):
    """
    Get ROC curve image
    
    Returns: Binary image file (PNG)
    """
    try:
        model_dir = os.path.join(settings.BASE_DIR, 'ml_models')
        roc_path = os.path.join(model_dir, 'roc_curve.png')

        if not os.path.exists(roc_path):
            return Response(
                {'error': 'ROC curve not found'},
                status=status.HTTP_404_NOT_FOUND
            )

        with open(roc_path, 'rb') as f:
            image_data = f.read()

        return Response(
            image_data,
            content_type='image/png',
            status=status.HTTP_200_OK
        )

    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def get_prediction_info(request):
    """
    Get prediction system information
    
    Response:
    {
        "success": true,
        "model_loaded": true,
        "direct_rules_count": 14,
        "risk_groups_count": 3,
        "system_status": "operational"
    }
    """
    try:
        service = NeonatalPredictionService()
        
        return Response(
            {
                'success': True,
                'model_loaded': service.model is not None,
                'direct_rules_count': len(service.DIRECT_RULES),
                'risk_groups_count': len(service.RISK_GROUPS),
                'system_status': 'operational',
                'methods': ['direct_rule', 'risk_group', 'ml_model']
            },
            status=status.HTTP_200_OK
        )

    except Exception as e:
        return Response(
            {
                'success': False,
                'error': str(e)
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
