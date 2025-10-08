from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from rest_framework.authtoken.models import Token
from django.contrib.auth.models import User
from .models import QA, ChatSession, ChatMessage
from .serializers import QASerializer, ChatSessionSerializer, ChatMessageSerializer
from .utils import find_best_match
import os, requests


class RegisterUser(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        username = request.data.get("username")
        password = request.data.get("password")

        if not username or not password:
            return Response({"error": "username and password required"}, status=400)

        if User.objects.filter(username=username).exists():
            return Response({"error": "User already exists"}, status=400)

        user = User.objects.create_user(username=username, password=password)
        token = Token.objects.create(user=user)
        return Response({"token": token.key}, status=201)


class ChatMessageView(APIView):
    """
    Endpoint to send a message and get a response
    POST /api/chat/message/
    """
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        message = request.data.get("message", "")
        session_id = request.data.get("session_id", "")
        
        if not message:
            return Response({"error": "Message is required"}, status=400)
        
        if not session_id:
            return Response({"error": "session_id is required"}, status=400)
        
        # Get or create session
        session, created = ChatSession.objects.get_or_create(
            session_id=session_id,
            defaults={'user': request.user if request.user.is_authenticated else None}
        )
        
        # Save user message
        user_msg = ChatMessage.objects.create(
            session=session,
            content=message,
            sender='user'
        )
        
        # Find best match from Q&A database
        qa_list = list(QA.objects.all())
        match = find_best_match(message, qa_list)
        
        if match:
            # Database match found
            response_text = match.answer
            source = "database"
            confidence = 0.8  # You can calculate actual confidence from utils.py
            
            # Save bot response
            bot_msg = ChatMessage.objects.create(
                session=session,
                content=response_text,
                sender='bot',
                confidence_score=confidence,
                source=source,
                matched_qa=match
            )
            
            return Response({
                "answer": response_text,
                "message": response_text,
                "source": source,
                "confidence": confidence,
                "matched_question": match.question if match else None
            })
        else:
            # No match, use AI
            ai_response = call_ai_agent(message)
            source = "ai"
            confidence = 0.6
            
            # Save bot response
            bot_msg = ChatMessage.objects.create(
                session=session,
                content=ai_response,
                sender='bot',
                confidence_score=confidence,
                source=source
            )
            
            return Response({
                "answer": ai_response,
                "message": ai_response,
                "source": source,
                "confidence": confidence
            })


class ChatHistoryView(APIView):
    """
    Endpoint to get chat history for a session
    GET /api/chat/history/?session_id=xxx
    """
    permission_classes = [permissions.AllowAny]
    
    def get(self, request):
        session_id = request.query_params.get('session_id', '')
        
        if not session_id:
            return Response({"error": "session_id is required"}, status=400)
        
        try:
            session = ChatSession.objects.get(session_id=session_id)
            serializer = ChatSessionSerializer(session)
            return Response(serializer.data)
        except ChatSession.DoesNotExist:
            # Return empty messages for new session
            return Response({
                "session_id": session_id,
                "messages": []
            })


def call_ai_agent(query):
    """
    Uses Anthropic Claude API for medical questions.
    Falls back to helpful message if API fails.
    """
    test_api_key = "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"
    
    if not test_api_key or test_api_key == "your-api-key-here":
        return "I'm a medical assistant chatbot. I can help answer questions about maternal health, pregnancy care, labour procedures, and more. Please add more questions to my knowledge base for better responses."

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": test_api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    # Add medical context to the system prompt
    system_prompt = """You are a specialized medical assistant focused on maternal health and obstetrics. 
    Provide accurate, evidence-based information about:
    - Pregnancy care and monitoring
    - Labour and delivery procedures
    - Postpartum care
    - Neonatal health
    - Medical protocols and guidelines
    
    Always be clear, professional, and remind users to consult with healthcare professionals for personalized medical advice."""
    
    data = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 500,
        "system": system_prompt,
        "messages": [{"role": "user", "content": query}]
    }

    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)
        response_data = response.json()
        
        if response.status_code == 200:
            return response_data.get("content", [{}])[0].get("text", "No response generated.")
        else:
            return "I apologize, but I'm having trouble processing your request. Please try rephrasing your question or contact a healthcare professional for urgent matters."
    except Exception as e:
        return f"I'm here to help with medical questions. However, I'm currently having connectivity issues. Please try again or consult with a healthcare professional."









"""

    curl https://api.anthropic.com/v1/messages \
        --header "test_api_key: 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" \
        --header "anthropic-version: 2023-06-01" \
        --header "content-type: application/json" \
        --data \
    '{
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Hello, world"}
        ]
    }'
"""