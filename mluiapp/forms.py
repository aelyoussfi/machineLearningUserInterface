from django import forms 
from . import models

class UiForm(forms.ModelForm):
    class Meta:
        model = models.inferenceJob
        fields = ['mlmodel', 'start_date', 'end_date']

class UiFormFitting(forms.ModelForm):
    class Meta:
        model = models.fittingJob
        fields = ['mlmodel_fitting']