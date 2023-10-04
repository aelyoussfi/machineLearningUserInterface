from django.contrib import admin
from . import models

# Register your models here.
admin.site.register(models.inferenceJob)
admin.site.register(models.fittingJob)