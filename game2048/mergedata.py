
import pandas as pd
from pandas import DataFrame as df
import numpy as np
import os

inputfile_dir='./inputfile'
outputfile='all2.csv'
for inputfile in os.listdir(inputfile_dir):
    inputfile1=os.path.join("./inputfile",inputfile)
    df = pd.read_csv(inputfile1)
    df.to_csv(outputfile,index=False, header=False, mode='a+')


'''
import glob
import time

csvx_list = glob.glob('*.csv')
print('总共发现%s个CSV文件'% len(csvx_list))
time.sleep(2)
print('正在处理............')
for i in csvx_list:
    fr = open(i,'r').read()
    with open('csv_to_csv.csv','a') as f:
        f.write(fr)'''