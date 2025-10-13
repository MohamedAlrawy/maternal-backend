from django.urls import path
from .views import ChatAPIView, ChatHistoryAPIView

urlpatterns = [
    path("chat/message/", ChatAPIView.as_view(), name="chat"),
    path("chat/history/", ChatHistoryAPIView.as_view(), name="chat-history"),
]
