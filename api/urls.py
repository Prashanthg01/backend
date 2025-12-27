from django.urls import path
from .views import (
    process_csv, get_filter_options, generate_schedule, 
    get_schedule, get_kpis, initialize_data,
    get_buffer_optimization, get_bottleneck_analysis,
    get_batch_optimization_preview
)

urlpatterns = [
    path('process-csv/', process_csv, name='process_csv'),
    path('get-filter-options/', get_filter_options, name='get_filter_options'),
    path('generate-schedule/', generate_schedule, name='generate_schedule'),
    path('get-schedule/', get_schedule, name='get_schedule'),
    path('get-kpis/', get_kpis, name='get_kpis'),
    path('initialize-data/', initialize_data, name='initialize_data'),
    path('buffer-optimization/', get_buffer_optimization, name='buffer_optimization'),
    path('bottleneck-analysis/', get_bottleneck_analysis, name='bottleneck_analysis'),
    path('batch-optimization-preview/', get_batch_optimization_preview, name='batch_optimization_preview'),
]