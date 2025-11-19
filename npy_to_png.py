# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 21:19:32 2025

@author: barbate
"""

import numpy as np
from PIL import Image
from PIL import ImageOps
import os
import matplotlib.pyplot as plt

path = r"C:\Users\barbate\OneDrive - Fraunhofer\new dataset\Needle detection Dataset\feta_oil"

count = 0
frame = 0

for filename in os.listdir(path):
    if filename.endswith(".npy"):
        file_path = os.path.join(path, filename)
        
        data = np.load(file_path, allow_pickle = True).item()
        
        #print(data)
        
        
        array = data['data'][0]#[0:480,0:500]
        # print('array->',array)
        # print(array.shape)
        
        array_min = np.min(array) 
        array_max = np.max(array) + 200
        #print('array_min',array_min)
        # print('array_max-->',array_max)
        
        
        norm_array1 = ((array - array_min) / (array_max - array_min))
        
        
        norm_array_uint16 = (norm_array1 * 65535).astype(np.uint16)
        # print('norm_array uint16->',norm_array_uint16[0])
        
        img = Image.fromarray(norm_array_uint16, mode='I;16')
        img.show()
        
        # img1 = Image.fromarray(array, mode='F')
        # img1.show()
        
        img.save(f"C:/Users/barbate/Roughwork/frameo.png")
        # frame += 1
       
     
        
        count += 1
        if count == 1:
            break
        

        
   # plt.imshow(array, cmap='gray', interpolation='nearest')
  
   
   # plt.axis("off")
   # plt.imshow()
   # plt.savefig("pixel_zoom.png")
   
   # array_min = np.min(array)
   # array_max = np.max(array)
   # print('array_min',array_min)
   # print('array_max-->',array_max)
   
   
   # norm_array1 = ((array - array_min) / (array_max - array_min))
   # print('norm_array1->',norm_array1)
   
   # norm_array = np.clip(norm_array1, 0, 1)
   # print('clip-->', norm_array)
   
   # arr_uint8 = (norm_array * 255).astype(np.uint8)
   
   # img = Image.fromarray(norm_array, mode='L')
   # #img.save("frame.png")

 
   # img.show() 

