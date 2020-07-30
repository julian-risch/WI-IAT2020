#%%
import pandas as pd
import csv

#%%
filename = '/mnt/data/datasets/newspapers/guardian/c_comments.csv'

#%%
header = ['article_id', 'comment_author_id', 'comment_id', 'comment_text', 'timestamp', 'parent_comment_id', 'upvotes']
header_dict = { el: i for i, el in enumerate(header)}

#%%
def parse_line(line):
    return list(csv.reader([line]))[0]

#%%
def get_comment_id(lin):


#%%
with open(filename, 'r') as f:
    f.seek(0)
    print(parse_line(f.readline()))
    print(f.tell())
    f.seek(0)
    print(f.tell())
    f.readline()
    print(f.tell())

#%%
with open(filename, 'r', encoding='utf-8') as f:
    f.readline() # move over header
    while True:
        offset = f.tell()
        line = f.readline()
        if not line:
            break
        comment_id = parse_line(line)[header_dict['comment_id']]
        print(comment_id)
        print(offset)



#%%
import subprocess
int(subprocess.check_output(["wc", "-l", filename]).split()[0])

#%%
