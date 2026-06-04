from django.urls import path

from . import train_views, views

urlpatterns = [
    path("health/", views.health, name="health"),
    path("model/upload/", views.upload_model, name="upload_model"),
    path("model/status/", views.model_status, name="model_status"),
    path("predict/", views.predict, name="predict"),
    path("predict/file/", views.predict_file, name="predict_file"),
    path("export/", views.export_csv, name="export_csv"),
    # Training
    path("train/defaults/", train_views.train_defaults, name="train_defaults"),
    path("train/dataset/upload/", train_views.upload_dataset, name="upload_dataset"),
    path("train/init-model/upload/", train_views.upload_init_model, name="upload_init_model"),
    path("train/start/", train_views.start_training, name="start_training"),
    path("train/jobs/<str:job_id>/", train_views.job_status, name="job_status"),
    path("train/jobs/<str:job_id>/model/", train_views.download_model, name="download_model"),
]
