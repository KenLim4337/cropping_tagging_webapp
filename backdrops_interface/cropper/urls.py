from django.urls import path

from . import views

app_name = 'cropper'

urlpatterns = [
    path('', views.home, name='home'),
]