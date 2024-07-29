# %%
import s3fs 
import os 
import numpy as np 
import pandas as pd 
# %%
goes = "18" 
bucket = f"noaa-goes{goes}" 
product = "ABI-L2-AODF" 
date_range = pd.date_range("2023-07-08", "2023-07-09", freq="h") 
s3 = s3fs.S3FileSystem(anon=True) 
# s3.ls(f"{bucket}") ## uncomment to list all goes products 
for doi in date_range: 
    print(doi.strftime("%Y-%m-%d-T%H")) 
    file = s3.ls( f"{bucket}/{product}/{doi.strftime('%Y')}/{doi.strftime('%j')}/{doi.strftime('%H')}" )[0] 
    file_name = file.split("/")[-1] 
    s3.download( file, f'/Users/jmceachern/data/{bucket}/{product}/{doi.strftime("%Y")}/{doi.strftime("%j")}/{file_name}')
 # %%
