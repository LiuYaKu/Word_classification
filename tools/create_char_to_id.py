# _*_ encoding:utf-8 _*_
__author__ = 'JQXX'
__date__ = '2018/11/5 23:55'
import os
os.chdir('../')
from config import FLAGS
m = open(FLAGS.data_path+"char_to_id.txt","w",encoding="utf-8")
z_m = open(FLAGS.data_path+"z_char_to_id.txt","w",encoding="utf-8")
r = open(FLAGS.data_path+'real_data.txt',"r",encoding='utf-8')
f = open(FLAGS.data_path+'feak_data.txt',"r",encoding='utf-8')
fn = open(FLAGS.data_path+"relu_no_proword.txt","r",encoding='utf-8')
fw = open(FLAGS.data_path+"test.txt","r",encoding='utf-8')

r_set=set()
for i in r:
    for j in i.strip():
        r_set.add(j)
for i in f:
    for j in i.strip():
        r_set.add(j)
for i in fn:
    for j in i.strip():
        r_set.add(j)
for i in fw:
    for j in i.strip():
        r_set.add(j)

for i,j in enumerate(r_set,1):
    m.write(j+" "+str(i)+"\n")
    z_m.write(j+" "+str(i)+"\n")

m.close()
z_m.close()
f.close()
r.close()