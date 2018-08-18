import zipfile
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array
from numpy import argmax
import sys
import os
import warnings
import pprint
import json

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

def OneHotConverterResult(oneHotData, training_set_dir):
    data = []
    for dir in sorted(os.listdir(training_set_dir)):
        data.append(dir)
    
    values = array(data)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
 
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
    
    with open('rules.json') as f :
        
        data = json.load(f)
        
        dicko = {}
        
        
        
        
        #print(onehot_encoded)
        for one_hot_encoder in onehot_encoded:
            currency_name = label_encoder.inverse_transform([argmax(one_hot_encoder)])
            #print(one_hot_encoder,  " -- " , currency_name)
            currency_details =data[currency_name[0]]
            #print(currency_details)
            #print(one_hot_encoder)
            #dicko[one_hot_encoder] = "y"
            
            #print(''.join(one_hot_encoder.tolist()))
            #print(''.join(one_hot_encoder))
            
            s = ""
            for letter_list in one_hot_encoder:
                letter = str(int(letter_list))
                s = s + ''.join(letter)
                
            
            print(s)
            dicko[s] = currency_details
            print('nn')
            
        
        
        
        print("XXXXxXXXXXXXXXXXXXXXXXXXxXXXXXXXXXXXXXXXXXXXxXXXXXXXXXXXXXXXXXXXxXXXXXXXXXXXXXXXXXXXxXXXXXXXXXXXXXXX")
        #print(dicko)
        print(json.dumps(dicko, indent=4))
        
        with open('result.json','w') as outfile:
            json.dump(dicko, outfile)
            
        print("Success")

        
    inverted = label_encoder.inverse_transform([argmax(oneHotData)])
    return inverted



data = OneHotConverterResult([0, 0, 1], "dataset/training_set/")
print(data)
