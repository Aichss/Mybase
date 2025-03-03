from django.shortcuts import get_object_or_404

from App import models
from App.models import Company
from LSTMPredictStock import run


def create_company(stock_code,name):
    return Company.objects.create(stock_code=stock_code,name=name)


