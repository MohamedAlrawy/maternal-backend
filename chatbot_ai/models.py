from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class QA(models.Model):
    question = models.TextField()
    answer = models.TextField()
    keywords = models.TextField(blank=True, null=True)
    category = models.CharField(max_length=100, blank=True, default='general')
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Medical Q&A"
        verbose_name_plural = "Medical Q&As"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.question[:50]}..."


class ChatSession(models.Model):
    session_id = models.CharField(max_length=255, unique=True, db_index=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"Session {self.session_id[:8]}..."


class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    content = models.TextField()
    sender = models.CharField(max_length=10, choices=[('user', 'User'), ('bot', 'Bot')])
    timestamp = models.DateTimeField(auto_now_add=True)
    confidence_score = models.FloatField(null=True, blank=True)
    source = models.CharField(max_length=20, null=True, blank=True)
    matched_qa = models.ForeignKey(QA, on_delete=models.SET_NULL, null=True, blank=True)
    
    class Meta:
        ordering = ['timestamp']
    
    def __str__(self):
        return f"{self.sender}: {self.content[:30]}..."
