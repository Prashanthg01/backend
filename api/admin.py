from django.contrib import admin
from .models import Product, Machine, ProcessStep, ProductionSchedule


# -------------------------
# Inline for Process Steps
# -------------------------
class ProcessStepInline(admin.TabularInline):
    model = ProcessStep
    extra = 1
    fields = (
        "step_number", "step_name", "machine",
        "cycle_time_seconds", "workers_required"
    )
    ordering = ("step_number",)


# -------------------------
# Product Admin
# -------------------------
@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = (
        "item", "sap_tn", "sap_pl", "dcc_type",
        "demand_2024", "batch_size", "num_batches",
        "created_at", "updated_at",
    )
    search_fields = ("item", "sap_tn", "sap_pl", "description")
    list_filter = ("dcc_type",)
    ordering = ("item",)
    
    inlines = [ProcessStepInline]


# -------------------------
# Machine Admin
# -------------------------
@admin.register(Machine)
class MachineAdmin(admin.ModelAdmin):
    list_display = ("name", "available_hours_per_day", "created_at", "updated_at")
    search_fields = ("name",)
    ordering = ("name",)


# python manage.py makemigrations 
# python manage.py migrate
# -------------------------
# Process Step Admin
# -------------------------
@admin.register(ProcessStep)
class ProcessStepAdmin(admin.ModelAdmin):
    list_display = (
        "product", "step_number", "step_name",
        "machine", "cycle_time_seconds",
        "workers_required", "created_at"
    )
    search_fields = ("step_name", "product__item")
    list_filter = ("machine", "product")
    ordering = ("product", "step_number")


# -------------------------
# Production Schedule Admin
# -------------------------
@admin.register(ProductionSchedule)
class ProductionScheduleAdmin(admin.ModelAdmin):
    list_display = (
        "batch_id", "batch_num", "batch_size",
        "machine", "product", "process_step",
        "start_time", "end_time", "duration_hours"
    )
    search_fields = ("batch_id", "machine__name", "product__item")
    list_filter = ("machine", "product")
    ordering = ("start_time",)

    # Helpful for time-based filtering
    date_hierarchy = "start_time"
