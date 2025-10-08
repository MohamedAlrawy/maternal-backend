from rest_framework import serializers
from .models import QA, ChatSession, ChatMessage

class QASerializer(serializers.ModelSerializer):
    class Meta:
        model = QA
        fields = '__all__'


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ['id', 'content', 'sender', 'timestamp', 'confidence_score', 'source']


class ChatSessionSerializer(serializers.ModelSerializer):
    messages = ChatMessageSerializer(many=True, read_only=True)
    
    class Meta:
        model = ChatSession
        fields = ['session_id', 'user', 'created_at', 'messages']
