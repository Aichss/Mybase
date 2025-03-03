import json
import os

import pandas as pd
from django.shortcuts import render, get_object_or_404

from App import models
from App.models import Company
from LSTMPredictStock import run


# Create your views here.
LOCAL = False


def get_hist_predict_data(stock_code):
    recent_data,predict_data = None,None
    #company = models.Company.objects.get(stock_code=stock_code)

    #Company.objects.create(stock_code=stock_code)
    company = get_object_or_404(Company, stock_code=stock_code)


    history_data = models.HistoryData()
    history_data.company = company
    history_data.set_data(run.get_hist_data(stock_code=stock_code,recent_day=20))
    history_data.save()
    recent_data = history_data.get_data()




    predict_data = models.PredictData()
    predict_data.company = company
    predict_data.set_data(run.prediction(stock_code,pre_len=10))
    predict_data.save()
    predict_data = predict_data.get_data()


    return recent_data,predict_data

def test(request):
    return render(request,"App/home.html")


def home(request):

    recent_data,predict_data = get_hist_predict_data("600839")
    data = {"recent_data":recent_data,"stock_code":"600839","predict_data":predict_data}

    return render(request,"App/home.html",{"data": json.dumps(data)}) # json.dumps(list)

def predict_stock_action(request):
    stock_code = request.POST.get('stock_code',None)
    print("stock_code:\n",stock_code)
    recent_data, predict_data = get_hist_predict_data(stock_code)
    data = {"recent_data": recent_data, "stock_code": stock_code, "predict_data": predict_data}

    return render(request, "App/home.html", {"data": json.dumps(data)})  # json.dumps(list)
