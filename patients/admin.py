from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User, Patient, ProgressNote, Alert


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ['username', 'email', 'first_name', 'last_name', 'role', 'is_active', 'date_joined']
    list_filter = ['role', 'is_active', 'is_staff', 'date_joined']
    search_fields = ['username', 'email', 'first_name', 'last_name']
    ordering = ['-date_joined']
    
    fieldsets = BaseUserAdmin.fieldsets + (
        ('Additional Info', {'fields': ('role', 'phone_number')}),
    )


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ['name', 'file_number', 'patient_id', 'age', 'case_type', 'room', 'stage', 'created_at']
    list_filter = ['case_type', 'stage', 'blood_group', 'booking', 'created_at']
    search_fields = ['name', 'file_number', 'patient_id', 'nationality']
    readonly_fields = ['bmi', 'created_at', 'updated_at']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'file_number', 'patient_id', 'nationality', 'age')
        }),
        ('Physical Measurements', {
            'fields': ('height', 'weight', 'bmi')
        }),
        ('Pregnancy Information', {
            'fields': ('gravidity', 'abortion', 'lmp', 'edd', 'gestational_age', 'booking')
        }),
        ('Medical Information', {
            'fields': ('blood_group',)
        }),
        ('Vital Signs', {
            'fields': ('pulse', 'bp_systolic', 'bp_diastolic', 'temp', 'oxygen_sat')
        }),
        ('Current Status', {
            'fields': ('case_type', 'room', 'stage', 'duration')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'created_by'),
            'classes': ('collapse',)
        }),
    )


@admin.register(ProgressNote)
class ProgressNoteAdmin(admin.ModelAdmin):
    list_display = ['patient', 'author', 'created_at']
    list_filter = ['created_at', 'author']
    search_fields = ['patient__name', 'note', 'author__username']
    readonly_fields = ['created_at']
    ordering = ['-created_at']


@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ['alert_type', 'patient', 'message_short', 'is_acknowledged', 'created_at']
    list_filter = ['alert_type', 'is_acknowledged', 'created_at']
    search_fields = ['message', 'patient__name']
    readonly_fields = ['created_at', 'acknowledged_at']
    ordering = ['-created_at']
    
    def message_short(self, obj):
        return obj.message[:50] + '...' if len(obj.message) > 50 else obj.message
    message_short.short_description = 'Message'