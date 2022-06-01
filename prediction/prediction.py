from pickle import load
import numpy as np
import pandas as pd
import warnings
import xgboost
import pkg_resources
warnings.filterwarnings("ignore")

def model_predict(X):
    
    cat_encoder = load(open(pkg_resources.resource_filename(__name__,'model/cat_encoder.pkl'),'rb'))
    model = load(open(pkg_resources.resource_filename(__name__,'model/model.pkl'),'rb'))
    scaler = load(open(pkg_resources.resource_filename(__name__,'model/scaler.pkl'),'rb'))
    X = np.asarray(X)
    one_hot_encode = cat_encoder.transform(X[0:5].reshape(1,-1))
    df_one_hot = pd.DataFrame(one_hot_encode.toarray())
    to_scale = scaler.transform(X[5:10].reshape(1,-1))
    df_scale = pd.DataFrame(to_scale)
    df_scale.rename(columns={0: 'Thickness', 1: 'Width',2: 'length',3: 'Bottom',4:'Quantity' },inplace=True)

    
    return model.predict(df_one_hot.join(df_scale))[0]

        

def unit_price_with_tax(X):
    """ Calculates unit price including tax. The default tax rate is 29%.""" 

    input_file = pd.read_csv(pkg_resources.resource_filename(__name__,'model/input_file.csv'))
    tax_rate =  input_file.iloc[0,0]
    unit_price = model_predict(X)
    return unit_price + tax_rate/100*unit_price


def total_unit_price_with_freight(X):
    """ Takes the dimension of size of plastic bag and quantity to estimate the total volume cbm and freight price.
        The default values are:
        zone1 = 10 
        cbm1 (<4cbm) = 217
        cbm2 (4cbm-6cbm) = 187
        cbm3 (6cbm-10cbm) = 172
        cbm4 (10cbm-15cbm) = 157
        cbm5 (15cbm-20cbm) = 142
        returns total unit price including tax and freight """
    
    length = X[7]/1000
    width = X[6]/1000
    thickness = 2*X[5]/1000000
    quantity = X[9]
    total_cbm = length*width*thickness*quantity

    if total_cbm < 1:
        total_cbm = 1

    input_file = pd.read_csv(pkg_resources.resource_filename(__name__,'model/input_file.csv'))

    if total_cbm <= 4:
        cbm = input_file.iloc[0,2]
    elif total_cbm > 4 and total_cbm <= 6:
        cbm = input_file.iloc[0,3]
    elif total_cbm > 6 and total_cbm <= 6:
        cbm = input_file.iloc[0,4]
    elif total_cbm > 10 and total_cbm <= 15:
        cbm = input_file.iloc[0,5]
    else:
        cbm = input_file.iloc[0,6]

    zone1 =  input_file.iloc[0,1]    

    freight = (cbm + zone1)*total_cbm
    unit_freight = freight/quantity

    return unit_price_with_tax(X)+unit_freight

def total_weight(X):
    """ Takes the dimension of size of plastic bag and quantity to estimate the total weight.
        returns total weight"""
    
    length = X[7]/1000
    width = X[6]/1000
    thickness = 0.04/100
    quantity = X[9]

    total_cbm = length*width*thickness*quantity

    if X[1] == 'PET/AL/PE' or X[1] == 'PET/AL/NY/PE' or X[1] == 'MOPP/AL/PE':
        density = 4800
    else:
        density = 2800

    total_kg = length*width*thickness*density*quantity

    return total_kg


def get_exw2(X):
     """ This function returns EXW price after multiplying by a factor which is determined based on 'Digital/Plate', Quantity and width."""
    
     print = X[3]
     wid = X[6]/25.4 # Change mm to inch by dividing 25.4
     quantity = X[9]

     if print == 'Digital':
         if wid < 4:
             if quantity <= 1000:
                 mu = 2.12
             else:
                 mu = 2
                 
         elif 4 <= wid < 5:
             if quantity <= 1000:
                 mu = 2
             else:
                 mu = 2.2
                 
         elif 5 <= wid < 7.5:
             if quantity <= 1000:
                 mu = 2.3
             elif 1000 < quantity <= 5000:
                 mu = 1.12
             elif 5000 < quantity <= 10000:
                 mu = 2
             else:
                 mu = 2.75
                 
         elif  7.5 <= wid < 9:
             if quantity <= 1000:
                 mu = 2.3
             elif 1000 < quantity <= 5000:
                 mu = 3
             elif 5000 < quantity <= 10000:
                 mu = 3
             elif 10000 < quantity <= 25000:
                 mu = 2.85
             elif 25000 < quantity <= 50000:
                 mu = 2.75
             else:
                 mu = 2.5
                 
         else :
             if quantity <= 1000:
                 mu = 1.5
             elif 1000 < quantity <= 5000:
                 mu = 4
             elif 5000 < quantity <= 10000:
                 mu = 3
             elif 10000 < quantity <= 25000:
                 mu = 3
             elif 25000 < quantity <= 50000:
                 mu = 3
             else:
                 mu = 2.75


     else :
         if wid < 4:
             if quantity <= 1000:
                 mu = 1.5
             elif 1000 < quantity <= 5000:
                 mu = 1.5
             elif 5000 < quantity <= 10000:
                 mu = 2.25
             elif 10000 < quantity <= 25000:
                 mu = 5
             elif 25000 < quantity <= 50000:
                 mu = 4.5
             else:
                 mu = 4
                 
         elif 4 <= wid < 5:
             if quantity <= 1000:
                 mu = 1.7
             elif 1000 < quantity <= 5000:
                 mu = 1.7
             elif 5000 < quantity <= 10000:
                 mu = 3
             elif 10000 < quantity <= 25000:
                 mu = 3.2
             elif 25000 < quantity <= 50000:
                 mu = 4.4
             else:
                 mu = 4
                 
         elif 5 <= wid < 7.5:
             if quantity <= 1000:
                 mu = 1.75
             elif 1000 < quantity <= 5000:
                 mu = 1.75
             elif 5000 < quantity <= 10000:
                 mu = 2
             elif 10000 < quantity <= 25000:
                 mu = 2
             elif 25000 < quantity <= 50000:
                 mu = 1.75
             else:
                 mu = 1.5
                 
         elif  7.5 <= wid < 9:
             if quantity <= 1000:
                 mu = 2
             elif 1000 < quantity <= 5000:
                 mu = 2
             elif 5000 < quantity <= 10000:
                 mu = 2.1
             elif 10000 < quantity <= 25000:
                 mu = 3
             elif 25000 < quantity <= 50000:
                 mu = 2.5
             else:
                 mu = 2.1
                 
         else :
             if quantity <= 1000:
                 mu = 2.5
             elif 1000 < quantity <= 5000:
                 mu = 2.5
             elif 5000 < quantity <= 10000:
                 mu = 2.5
             elif 10000 < quantity <= 25000:
                 mu = 2.5
             elif 25000 < quantity <= 50000:
                 mu = 2.5
             else:
                 mu = 2.5


     unit_price = model_predict(X)

     return unit_price*mu
        


      
    
