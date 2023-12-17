

from django.db import models

class rent_prediction(models.Model):
    BHK = models.IntegerField()
    Size = models.FloatField()
    Floor = models.IntegerField()
    Bathroom = models.IntegerField()
    FurnishingStatus = models.CharField(max_length=100)
    AreaType = models.CharField(max_length=100)
    AreaLocality = models.CharField(max_length=200)
    City = models.CharField(max_length=100)
    PredictedRent = models.FloatField()

    class Meta:
        db_table = 'rent_prediction'


class milk_prediction(models.Model):
    pH = models.FloatField()
    Temprature = models.FloatField()
    Taste = models.FloatField()
    Odor = models.FloatField()
    Fat = models.FloatField()
    Turbidity = models.FloatField()
    Colour = models.FloatField()
    milkquality = models.CharField(max_length=10)

    class Meta:
        db_table = 'milk_prediction'