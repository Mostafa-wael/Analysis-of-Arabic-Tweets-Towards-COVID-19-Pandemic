# This script extracts the arabic text from the dataset and saves it in a text file

import pandas as pd

train_df = pd.read_csv('Dataset/train.csv', encoding='utf-8')
dev_df = pd.read_csv('Dataset/dev.csv', encoding='utf-8')

train_text = train_df['text'].values
dev_text = dev_df['text'].values



train_text_file = open('out/train_text.txt', 'w', encoding='utf-8')
dev_text_file = open('out/dev_text.txt', 'w', encoding='utf-8')


for i in range(len(train_text)):
    x = train_text[i]
    
    if '\n' in x:
        x = x.replace('\n', ' ')

    train_text_file.write(x)
    
    if i != len(train_text) - 1:
        train_text_file.write('\n')

for i in range(len(dev_text)):
    x = dev_text[i]

    if '\n' in x:
        x = x.replace('\n', ' ')
    dev_text_file.write(x)
    
    if i != len(dev_text) - 1:
        dev_text_file.write('\n')