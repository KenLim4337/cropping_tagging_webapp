from django.urls import path

from . import views
from . import api

app_name = 'cropper'

urlpatterns = [
    path('', views.home, name='home'),
    path('api/search/', api.get_results, name="get_results"),
]