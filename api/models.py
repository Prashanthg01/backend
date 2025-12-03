from django.db import models

class Product(models.Model):
    item = models.IntegerField(unique=True)
    sap_tn = models.CharField(max_length=50)
    sap_pl = models.CharField(max_length=50, null=True, blank=True)
    dcc_type = models.CharField(max_length=100)
    description = models.TextField()
    demand_2024 = models.IntegerField(default=0)
    batch_size = models.IntegerField(default=0)
    num_batches = models.IntegerField(default=12)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'products'
        ordering = ['item']
    
    def __str__(self):
        return f"Item {self.item} - {self.description}"


class Machine(models.Model):
    name = models.CharField(max_length=100, unique=True)
    available_hours_per_day = models.FloatField(default=24)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'machines'
        ordering = ['name']
    
    def __str__(self):
        return self.name


class ProcessStep(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='process_steps')
    step_number = models.IntegerField()
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name='process_steps')
    step_name = models.CharField(max_length=200)
    cycle_time_seconds = models.FloatField()
    workers_required = models.FloatField(default=0.5)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'process_steps'
        ordering = ['product', 'step_number']
        unique_together = ['product', 'step_number']
    
    def __str__(self):
        return f"{self.product.item} - Step {self.step_number}"


class ProductionSchedule(models.Model):
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name='schedules')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='schedules')
    process_step = models.ForeignKey(ProcessStep, on_delete=models.CASCADE, related_name='schedules')
    
    batch_id = models.CharField(max_length=50)
    batch_num = models.IntegerField()
    batch_size = models.IntegerField()
    
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    duration_hours = models.FloatField()
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'production_schedules'
        ordering = ['start_time', 'machine']
        indexes = [
            models.Index(fields=['start_time', 'end_time']),
            models.Index(fields=['machine', 'start_time']),
        ]
    
    def __str__(self):
        return f"{self.batch_id} - {self.machine.name}"