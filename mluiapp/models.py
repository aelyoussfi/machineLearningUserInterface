from django.db import models
from jsonfield import JSONField

# Create your models here.

class inferenceJob(models.Model):
    mlmodel = models.CharField(max_length=50)
    start_date = models.CharField(max_length=50)
    end_date = models.CharField(max_length=50)
    data = JSONField()
    score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

class fittingJob(models.Model):
    mlmodel_fitting = models.CharField(max_length=50)
    T = models.FloatField()
    train_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    