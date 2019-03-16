import os
os.chdir('../')
from config import FLAGS
test_file_name=FLAGS.test_data_path.split("/")[-1]
#f=open(FLAGS.result_path+test_file_name+"_result.txt",encoding="utf-8")
f = open("./data/real_data.txt",encoding="utf-8")
#fr=open("./result/new_relu_no_proword.txt_result.txt",encoding="utf-8")
fr = open(FLAGS.test_data_path,encoding="utf-8")
s=set()
for i in fr:
    s.add(i.strip())
st=set()
for i in f:
    i=i.strip()
    ban=int(len(i)/2)
    if i[:ban]==i[ban:]:
        print(i)
        continue
    st.add(i)
if len(st)==0:
    print("OK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    exit(0)
fr.close()
f.close()
fr=open(FLAGS.test_data_path,"w",encoding="utf-8")
for i in s:
    if i not in st:
        fr.write(i+"\n")
