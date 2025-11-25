# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 15:19:33 2025

@author: barbate
"""
import os
import pandas as pd
from PIL import Image, ImageDraw 


df = pd.read_csv(r"C:\Users\barbate\Downloads\labels_test_dataset.csv")

# print(df.columns)

# label_name = df["label_name"]
# print(label_name)

df["image_name"] = df["image_name"]
df["class_id"] = df["label_name"]

df["x_min"] = df["bbox_x"]
# print(bbox_x)

df["y_min"] = df["bbox_y"]
# print(bbox_y)

df["x_max"] = df["bbox_x"] + df["bbox_width"]
# print(bbox_width)

df["y_max"] = df["bbox_y"] + df["bbox_height"]
#print(bbox_height)

for index, row in df.iterrows():
    print(row["image_name"],row["class_id"],row["x_min"], row["y_min"], row["x_max"], row["y_max"])
    output_df = df[["image_name","class_id","x_min","y_min","x_max","y_max"]]
    output_path = r"C:\Users\barbate\Roughwork\good_frames_test"
    output_df.to_csv(output_path, index=False)
images = []
folder_path = r"C:\Users\barbate\Roughwork\good_frames_test\labels_output.csv"
for filename in os.listdir(folder_path):
    img_path = os.path.join(folder_path, filename)
    print(filename)
    img = Image.open(img_path)
    images.append(img)
    if filename in df["image_name"].values :
        print({filename})
        for _, row in df[df["image_name"] == filename].iterrows():
            draw = ImageDraw.Draw(img)
            draw.rectangle([row["x_min"], row["y_min"] , row["x_max"], row["y_max"]],  outline="red", width=3)
            draw.text((row["x_min"], row["y_min"]-10), row["class_id"], fill="red")
        img.show()
    

    
    


  

  
# y_min = bbox_y
# print("y_min:",y_min)
  
# x_max = bbox_x + bbox_width
# print("x_max:",x_max)
  
# y_max = bbox_y + bbox_height 
# print("y_max:",y_max)

# image_name = df["image_name"]
# print(image_name)

# image_width = df["image_width"]
# print(image_width)

# image_height = df["image_height"]
# print(image_height)

# for index, row in df.iloc[12:13].iterrows():
#     print(row["image_name"],
#           row["label_name"], 
#           row["bbox_x"], 
#           row["bbox_y"], 
#           row["bbox_height"], 
#           row["bbox_width"],
#           row["image_height"],
#           row["image_width"])
    
#     bbox_x = row["bbox_x"] 
#     bbox_y = row["bbox_y"]
#     bbox_width = row["bbox_width"]
#     bbox_height = row["bbox_height"]
    
#     x_min = bbox_x
#     print("x_min:",x_min)
    
#     bbox_y = row["bbox_y"]
#     bbox_height = row["bbox_height"]
    
#     y_min = bbox_y
#     print("y_min:",y_min)
    
#     x_max = bbox_x + bbox_width
#     print("x_max:",x_max)
    
#     y_max = bbox_y + bbox_height 
#     print("y_max:",y_max)
    
    
#     img = Image.open(r"C:\Users\barbate\Roughwork\good_frames_test\frame_825.png")
    
   
    
#     draw = ImageDraw.Draw(img)
#     draw.rectangle([x_min, y_min, x_max , y_max], outline="red", width=3)
    
#     img.show()
    
    
    
    


