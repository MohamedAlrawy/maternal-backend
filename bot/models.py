from django.db import models
from django.contrib.postgres.fields import ArrayField
import json
import uuid


class ChatSession(models.Model):
    """
    Represents a persistent chat session (can map to frontend session_uuid or Django session_key)
    """
    session_id = models.CharField(max_length=100, unique=True)  # can store request.session.session_key or UI UUID
    created_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"ChatSession {self.session_id}"


class ChatMessage(models.Model):
    """
    Stores individual chat turns
    """
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    sender = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    source = models.CharField(max_length=50, null=True, blank=True)  # e.g. 'faq' or 'ai'
    matched_section = models.CharField(max_length=200, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.role}: {self.text[:40]}..."

class Section(models.Model):
    header = models.CharField(max_length=500)
    context = models.TextField(help_text="The section context / explanation.")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.header

class QAPair(models.Model):
    section = models.ForeignKey(Section, related_name="qa_pairs", on_delete=models.CASCADE)
    question = models.TextField()
    answer = models.TextField()
    # store embedding as JSON text (list of floats)
    embedding = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def set_embedding(self, emb: list):
        self.embedding = emb

    def get_embedding(self):
        return self.embedding or []
