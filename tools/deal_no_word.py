import os
from config import FLAGS
import re
os.chdir('../')
s=set()
f_d=open(FLAGS.data_path+"relu_no_proword.txt","r",encoding="utf-8")
for i in f_d:
    s.add(i.strip())
f_d.close()
files = os.listdir(FLAGS.data_path + "book/")
all_sen = []
for path in files:
    f = open(FLAGS.data_path + "book/" + path, "r", encoding='utf-8')
    paragraph = "".join([m for m in f.readlines() if m.strip()])
    sentences = re.split('(。|！|\!|\.|？|\?)', paragraph)
    for i in range(int(len(sentences) / 2)):
        sent = sentences[2 * i] + sentences[2 * i + 1]
        sent = sent.replace("\n", "")
        sent = sent.replace("\t", "")
        sent = sent.replace(" ", "")
        if sent:
            all_sen.append(sent)
f_d=open(FLAGS.data_path+"new_relu_no_proword.txt","w",encoding="utf-8")
for i in s:
    is_find=False
    for sen in all_sen:
        if i in sen:
            is_find=True
            break
    if is_find:
        f_d.write(i+"\n")
f_d.close()