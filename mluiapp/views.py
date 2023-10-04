from django.shortcuts import render, redirect
from . import forms , models
from django.core.files import File as DjangoFile
from django.conf import settings
import numpy as np
import json
from brain import ml, MLFactory


# Create your views here.
def home(request):
    return render(request, 'mluiapp/home.html')

def inference(request):
    uiform_form = forms.UiForm
    ml_fact  = MLFactory.MLFact()
    if request.method == 'POST':
        uiform_form = forms.UiForm(request.POST)
        if uiform_form.is_valid():
            mlmodel = uiform_form.cleaned_data['mlmodel']
            startDate = uiform_form.cleaned_data['start_date']
            endDate = uiform_form.cleaned_data['end_date']

            # call brain work 
            s_i,e_i = ml_fact.get_interval('brain/train.csv')
            # x_train, x_test, y_train, y_test,test_X = ml.datagenerator('brain/train.csv','brain/test.csv',startDate,endDate)
            # train_score,test_score,T = ml.test_models(mlmodel,x_train,y_train,x_test,y_test)
            # myjob = models.Job.objects.create(mymodel = models_dict.keys)
            #jsondata = json.dumps(data)
            # myjob.save()
            #return redirect('showroom')
            data , score = ml_fact.inference(mlmodel,startDate,endDate)
            data = json.dumps(data)
            myinferencejob = models.inferenceJob.objects.create(mlmodel = mlmodel,score = score,data = data)
            myinferencejob.save()
            return redirect(inference_result)
    models_dict = ml_fact.get_models_list()
    models_names= ""
    models_val = []
    for k,v in models_dict.items():
        models_names += "*"+k+" "
        models_val.append(v)
    models_dict = models_names
    s_i,e_i = ml.get_interval('brain/train.csv')
    return render(request, 'mluiapp/inference.html', context={'form': uiform_form, 'dict': models_dict,'s_i': s_i, 'e_i': e_i})

def fitting(request):
    uiformFitting_form = forms.UiFormFitting
    ml_fact  = MLFactory.MLFact()
    message = ""
    x_train, y_train = ml_fact.x_train, ml_fact.y_train
    if request.method == 'POST':
        uiformFitting_form = forms.UiFormFitting(request.POST)
        if uiformFitting_form.is_valid():
            mlmodel_fitting = uiformFitting_form.cleaned_data['mlmodel_fitting']
            T,train_score = ml_fact.fitting(mlmodel_fitting,x_train, y_train)
            message = "Successfully fitting"

    models_dict = ml_fact.get_models_list()
    models_names= ""
    models_val = []
    for k,v in models_dict.items():
        models_names += "*"+k+" "
        models_val.append(v)
    models_dict = models_names
    return render(request, 'mluiapp/fitting.html', context={'form': uiformFitting_form,'dict_models':models_dict,"message":message})
            
def inference_result(request):
    Results = models.inferenceJob.objects.order_by('-created_at')[:1]
    return render(request, 'mluiapp/inference_result.html', context={'results':Results})