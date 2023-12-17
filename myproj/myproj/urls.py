from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),  #
    path('index/', views.index, name='index'),  # Home page
    # path('milk_analysis/', views.milk_analysis, name='milk_analysis'),
    # path('milk_quality_view/', views.milk_quality_view, ),
    path('classification/', views.classification, name='classification'),  # Classification
    path('regression/', views.regression, name='regression'), #Regression
    path('predict_rent/', views.predict_rent, name='predict_rent'),  # URL for rent prediction
    path('predict_milk/', views.predict_milk, name='predict_milk'), # URL for milk prediction
    path('show/', views.show_records, name='show_records'),
    path('showreg/', views.show_rent, name='show_rent'),
    path('index/show/', views.show_records, name='show_records'),  
    path('index/showreg/', views.show_rent, name='show_rent'),  
]
