import pandas as pd
import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

path=input("Enter attack csv file name")

ds1=pd.read_csv(path)

summarizer = pipeline("summarization")

list1=ds1.perturbed_text
list2=[]


# Summarizer module
print("Running summarizer")
for x in tqdm(list1):
  art_len=len(x.split())
  if(len(x)>1024):
    x=x[:1024]
  list2.append(summarizer(x, max_length=art_len + 1, min_length=min(15,art_len//4), do_sample=False)[0]['summary_text'])
  

summarized=list2
perturbed=list1
initial_prob=list(ds1.original_output)
initial_score=list(ds1.original_score)
perturbed_prob=list(ds1.perturbed_output)
perturbed_score=list(ds1.perturbed_score)

final_prob=[]
final_score=[]

# Classifier module


tokenizer = AutoTokenizer.from_pretrained("VictorSanh/roberta-base-finetuned-yelp-polarity")

model = AutoModelForSequenceClassification.from_pretrained("VictorSanh/roberta-base-finetuned-yelp-polarity")

print("Running classifier")
for y in tqdm(summarized):
  txf1=tokenizer.encode_plus(y,return_tensors="pt",max_length=512)
  txf2=model(**txf1)[0]
  results=torch.softmax(txf2, dim=1).tolist()[0]
  final_prob.append(np.argmax(results))
  final_score.append(results[np.argmax(results)])

df = pd.DataFrame(list(zip(initial_prob,initial_score,perturbed_prob,perturbed_score,final_prob,final_score,summarized)), 
               columns =['Initial_P', 'Initial_S','Pert_P','Pert_S','Final_P','Final_S','Summarized']) 

s=0
len_ds1=len(list1)
for x in range(len_ds1):
  if(df.loc[x,'Initial_P']==df.loc[x,'Final_P']):
    s+=1

print(f"Percentage of successful attacks is {s/len(list1)*100}")
save_file = input("Enter Result file name")
df.to_csv(save_file)
