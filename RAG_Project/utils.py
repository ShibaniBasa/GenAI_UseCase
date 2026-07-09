# See chat for full commented version.
import os,pandas as pd
from langchain_core.documents import Document

def load_excel_file(p): 
    return pd.read_excel(p,sheet_name=None)

def row_to_paragraph(row,lang,sheet):
 s=[f'This information belongs to the {lang} dataset.',f'It comes from the {sheet} sheet.']
 [s.append(f'{c} is {str(row[c]).strip()}.') for c in row.index if pd.notna(row[c]) and str(row[c]).strip()]
 return ' '.join(s)

def create_documents(path):
 docs=[]
 data=load_excel_file(path)
 n=os.path.basename(path)                 #Detect language from filename
 lang='Python' if 'python' in n.lower() else 'Java' if 'java' in n.lower() else 'Unknown'
 from langchain_core.documents import Document
 for sh,df in data.items():
  for i,r in df.fillna('').iterrows(): 
     docs.append(Document(page_content=row_to_paragraph(r,lang,sh),metadata={'language':lang,'sheet_name':sh,'row_number':i+1,'source_file':n}))
 return docs

def load_all_documents(folder):
 a=[]
 [a.extend(create_documents(os.path.join(folder,f))) for f in os.listdir(folder) if f.endswith('.xlsx')]
 return a
