from django.urls import path
from .views import process_csv, get_filter_options

urlpatterns = [
    path('process-csv/', process_csv, name='process_csv'),
    path('get-filter-options/', get_filter_options, name='get-filter-options'),
]
