import os
os.chdir('../')
from config import FLAGS
test_file_name=FLAGS.test_data_path.split("/")[-1]
if not FLAGS.is_domain:
    test_file_name=test_file_name[2:]
f=open(FLAGS.result_path+test_file_name+"_result.txt",encoding="utf-8")
f1=open(FLAGS.data_path+"real_data.txt",encoding="utf-8")
f_w=open(FLAGS.result_path+test_file_name+"_unique_result.txt","w",encoding="utf-8")

s=set()
s1=set()
for i in f:
    s.add(i.strip())

for i in f1:
    s1.add(i.strip())

for i in s:
    #if i not in s1 and len(i)>4:
    f_w.write(i+"\n")

