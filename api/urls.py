# from django.urls import path
# from .views import (
#     get_task_status, process_csv, get_filter_options, generate_schedule,
#     get_schedule, get_kpis, initialize_data,
#     get_buffer_optimization, get_bottleneck_analysis,
#     # Batch optimization (individual)
#     get_batch_optimization_preview,
#     save_batch_optimization,
#     # Batch optimization (joint) — NEW
#     get_joint_batch_preview,
#     save_joint_batch_optimization,
#     get_schedule_gantt,
#     gap_analysis,
#     schedule_comparison,
#     optimize_schedule_preview,
#     optimize_schedule_save,
# )
# from .auth_views import login, logout, me

# urlpatterns = [
#     # Auth
#     path('auth/login/',  login,  name='auth_login'),
#     path('auth/logout/', logout, name='auth_logout'),
#     path('auth/me/',     me,     name='auth_me'),

#     path('process-csv/',         process_csv,        name='process_csv'),
#     path('get-filter-options/',  get_filter_options, name='get_filter_options'),
#     path('kpis/',                get_kpis,           name='get_kpis'),
#     path('initialize-data/',     initialize_data,    name='initialize_data'),
#     path('buffer-optimization/', get_buffer_optimization,  name='buffer_optimization'),
#     path('bottleneck-analysis/', get_bottleneck_analysis,  name='bottleneck_analysis'),

#     # Batch Optimization — individual (adaptive bounds fix applied in utils.py)
#     path('batch-optimization-preview/', get_batch_optimization_preview, name='batch_optimization_preview'),
#     path('batch-optimization-save/',    save_batch_optimization,        name='batch_optimization_save'),

#     # Batch Optimization — joint multi-product (NEW)
#     path('batch-optimization-joint-preview/', get_joint_batch_preview,       name='batch_optimization_joint_preview'),
#     path('batch-optimization-joint/',         save_joint_batch_optimization, name='batch_optimization_joint_save'),

#     path('task-status/<str:task_id>/', get_task_status, name='task_status'),

#     # Schedule Management
#     path('generate-schedule/',       generate_schedule,       name='generate_schedule'),
#     path('get-schedule/',            get_schedule,            name='get_schedule'),
#     path('schedule-gantt/',          get_schedule_gantt,      name='schedule_gantt'),
#     path('gap-analysis/',            gap_analysis,            name='gap_analysis'),
#     path('schedule-comparison/',     schedule_comparison,     name='schedule_comparison'),
#     path('optimize-schedule-preview/', optimize_schedule_preview, name='optimize_schedule_preview'),
#     path('optimize-schedule/',       optimize_schedule_save,  name='optimize_schedule_save'),
# ]


from django.urls import path
from .views import (
    get_task_status, process_csv, get_filter_options, generate_schedule,
    get_schedule, get_kpis, initialize_data,
    get_buffer_optimization, get_bottleneck_analysis,
    get_batch_optimization_preview,
    save_batch_optimization,
    get_joint_batch_preview,
    save_joint_batch_optimization,
    get_schedule_gantt,
    gap_analysis,
    schedule_comparison,
    optimize_schedule_preview,
    optimize_schedule_save,
    # ── Synthetic data ───────────────────────────────────────────
    initialize_synthetic_data,
    get_scenario_profiles,
)
from .auth_views import login, logout, me

urlpatterns = [
    # Auth
    path('auth/login/',  login,  name='auth_login'),
    path('auth/logout/', logout, name='auth_logout'),
    path('auth/me/',     me,     name='auth_me'),

    path('process-csv/',         process_csv,        name='process_csv'),
    path('get-filter-options/',  get_filter_options, name='get_filter_options'),
    path('kpis/',                get_kpis,           name='get_kpis'),
    path('initialize-data/',     initialize_data,    name='initialize_data'),
    path('buffer-optimization/', get_buffer_optimization,  name='buffer_optimization'),
    path('bottleneck-analysis/', get_bottleneck_analysis,  name='bottleneck_analysis'),

    # Batch Optimization — individual
    path('batch-optimization-preview/', get_batch_optimization_preview, name='batch_optimization_preview'),
    path('batch-optimization-save/',    save_batch_optimization,        name='batch_optimization_save'),

    # Batch Optimization — joint multi-product
    path('batch-optimization-joint-preview/', get_joint_batch_preview,       name='batch_optimization_joint_preview'),
    path('batch-optimization-joint/',         save_joint_batch_optimization, name='batch_optimization_joint_save'),

    path('task-status/<str:task_id>/', get_task_status, name='task_status'),

    # Schedule Management
    path('generate-schedule/',         generate_schedule,         name='generate_schedule'),
    path('get-schedule/',              get_schedule,              name='get_schedule'),
    path('schedule-gantt/',            get_schedule_gantt,        name='schedule_gantt'),
    path('gap-analysis/',              gap_analysis,              name='gap_analysis'),
    path('schedule-comparison/',       schedule_comparison,       name='schedule_comparison'),
    path('optimize-schedule-preview/', optimize_schedule_preview, name='optimize_schedule_preview'),
    path('optimize-schedule/',         optimize_schedule_save,    name='optimize_schedule_save'),

    # ── Synthetic data initialization & scenario profiles ────────────────────
    path('initialize-synthetic/', initialize_synthetic_data, name='initialize_synthetic'),
    path('scenario-profiles/',    get_scenario_profiles,     name='scenario_profiles'),
]