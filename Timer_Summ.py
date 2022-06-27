import pandas as pd
import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import time
from tqdm import tqdm

path=input("Enter attack csv file name")

ds1=pd.read_csv(path)

summarizer = pipeline("summarization")


tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")

model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")

time_sum=[]
time_cls=[]


list1=ds1.perturbed_text
list2=[]


# Summarizer module
print("Running defense")
for x in tqdm(list1):
  cnta=time.timeit()
  art_len=len(x.split())
  if(len(x)>1024):
    x=x[:1024]
  list2.append(summarizer(x, max_length=art_len + 1, min_length=min(15,art_len//4), do_sample=False)[0]['summary_text'])
  cntb=time.timeit()
  time_sum.append(cntb-cnta)
  a+=1


summarized=list2
perturbed=list1
initial_prob=list(ds1.original_output)
initial_score=list(ds1.original_score)
perturbed_prob=list(ds1.perturbed_output)
perturbed_score=list(ds1.perturbed_score)

final_prob=[]
final_score=[]

print("Running classifier over perturbed text")
for y in tqdm(summarized):
  cnta= time.timeit()
  txf1=tokenizer.encode_plus(y,return_tensors="pt",max_length=512)
  txf2=model(**txf1)[0]
  results=torch.softmax(txf2, dim=1).tolist()[0]
  final_prob.append(np.argmax(results))
  final_score.append(results[np.argmax(results)])
  cntb=time.timeit()
  time_sum[a]=time_sum[a]+(cntb-cnta)
  a+=1

print("Running classifier over input text")
a=0
for y in tqdm(perturbed):
  cnta= time.timeit()
  txf1=tokenizer.encode_plus(y,return_tensors="pt",max_length=512)
  txf2=model(**txf1)[0]
  results=torch.softmax(txf2, dim=1).tolist()[0]
  final_prob.append(np.argmax(results))
  final_score.append(results[np.argmax(results)])
  cntb=time.timeit()
  time_cls.append(cntb-cnta)
  a+=1


print(f"Average time w/o defense={np.mean(time_cls)}")
print(f"Average time w defense={np.mean(time_sum)}")
