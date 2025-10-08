from django.contrib import admin
from .models import QA, ChatSession, ChatMessage


@admin.register(QA)
class QAAdmin(admin.ModelAdmin):
    list_display = ['question_preview', 'category', 'created_at']
    list_filter = ['category', 'created_at']
    search_fields = ['question', 'answer', 'keywords']
    ordering = ['-created_at']
    
    def question_preview(self, obj):
        return obj.question[:80] + '...' if len(obj.question) > 80 else obj.question
    question_preview.short_description = 'Question'


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ['session_id_preview', 'user', 'created_at', 'message_count']
    list_filter = ['created_at']
    search_fields = ['session_id', 'user__username']
    readonly_fields = ['session_id', 'created_at', 'updated_at']
    ordering = ['-updated_at']
    
    def session_id_preview(self, obj):
        return obj.session_id[:20] + '...' if len(obj.session_id) > 20 else obj.session_id
    session_id_preview.short_description = 'Session ID'
    
    def message_count(self, obj):
        return obj.messages.count()
    message_count.short_description = 'Messages'


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['content_preview', 'sender', 'source', 'confidence_score', 'timestamp']
    list_filter = ['sender', 'source', 'timestamp']
    search_fields = ['content']
    readonly_fields = ['timestamp']
    ordering = ['-timestamp']
    
    def content_preview(self, obj):
        return obj.content[:100] + '...' if len(obj.content) > 100 else obj.content
    content_preview.short_description = 'Content'
