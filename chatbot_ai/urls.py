from django.urls import path
from .views import ChatMessageView, ChatHistoryView, RegisterUser

urlpatterns = [
    path('register/', RegisterUser.as_view(), name='register'),
    path('chat/message/', ChatMessageView.as_view(), name='chat-message'),
    path('chat/history/', ChatHistoryView.as_view(), name='chat-history'),
]
