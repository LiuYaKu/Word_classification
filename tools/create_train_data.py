# -*- coding: utf-8 -*-
"""
# @File  : create_train_data.py
# @Author: JQXX
# @Date  : 2018/12/1
"""
import os
os.chdir('../')
from config import FLAGS

f1 = open(FLAGS.data_path+"real_data.txt","r",encoding='utf-8')
f2 = open(FLAGS.data_path+"feak_data.txt","r",encoding='utf-8')
f3 = open(FLAGS.data_path+"relu_no_proword.txt","r",encoding='utf-8')
he_f = open(FLAGS.data_path+"train.txt","w",encoding='utf-8')

for lin in f1:
    he_f.write(lin.strip()+" 1"+"\n")
if not FLAGS.is_domain:
    for lin in f2:
        he_f.write(lin.strip()+" 1"+"\n")
else:
    for lin in f2:
        he_f.write(lin.strip()+" 0"+"\n")
for lin in f3:
    he_f.write(lin.strip()+" 0"+"\n")

f1.close()
f2.close()
f3.close()
he_f.close()