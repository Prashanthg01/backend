from rest_framework import serializers
from .models import Product, Machine, ProcessStep, ProductionSchedule

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = '__all__'


class MachineSerializer(serializers.ModelSerializer):
    class Meta:
        model = Machine
        fields = '__all__'


class ProcessStepSerializer(serializers.ModelSerializer):
    machine_name = serializers.CharField(source='machine.name', read_only=True)
    product_item = serializers.IntegerField(source='product.item', read_only=True)
    
    class Meta:
        model = ProcessStep
        fields = '__all__'


class ProductionScheduleSerializer(serializers.ModelSerializer):
    machine_name = serializers.CharField(source='machine.name', read_only=True)
    product_item = serializers.IntegerField(source='product.item', read_only=True)
    product_sap_tn = serializers.CharField(source='product.sap_tn', read_only=True)
    product_sap_pl = serializers.CharField(source='product.sap_pl', read_only=True)
    product_dcc_type = serializers.CharField(source='product.dcc_type', read_only=True)
    product_description = serializers.CharField(source='product.description', read_only=True)
    step_number = serializers.IntegerField(source='process_step.step_number', read_only=True)
    step_name = serializers.CharField(source='process_step.step_name', read_only=True)
    workers_required = serializers.FloatField(source='process_step.workers_required', read_only=True)
    cycle_time_seconds = serializers.FloatField(source='process_step.cycle_time_seconds', read_only=True)
    
    class Meta:
        model = ProductionSchedule
        fields = '__all__'