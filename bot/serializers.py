from rest_framework import serializers
from .models import Section, QAPair

class QAPairSerializer(serializers.ModelSerializer):
    section_header = serializers.CharField(source='section.header', read_only=True)
    class Meta:
        model = QAPair
        fields = ['id','section','section_header','question','answer']

class SectionSerializer(serializers.ModelSerializer):
    qa_pairs = QAPairSerializer(many=True, read_only=True)
    class Meta:
        model = Section
        fields = ['id','header','context','qa_pairs']
