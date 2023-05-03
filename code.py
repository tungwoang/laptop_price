

import pandas as pd
import numpy as np
import re
df=pd.read_csv('laptops_origin.csv')
pd.set_option('display.max_rows',None,'display.max_columns',None) 


# price india to VND

df['Price']=df['Price']*285/100000
df['Price']=df['Price'].astype('float')
# fix name

df['Model_Name1']=df['Model Name'].apply(lambda x : x.split(' ')[0])

df['Model_Name2']=df['Model Name'].apply(lambda x : x.split(' ')[1])

df['model_dl']=df['Model_Name2'].apply(lambda x : '' if x.startswith('(') or x.endswith('"') else x)

df['name']=df.apply(lambda x : x['Model_Name1']+' '+x['model_dl'],axis =1)

df['Model Name']=df.apply(lambda x : x['name'], axis =1 )

df.drop(['Model_Name1','Model_Name2', 'model_dl', 'name'] ,axis = 1 , inplace = True)

# Screen size

df['Screen Size']=df['Screen Size'].apply(lambda x : float(x.replace('"','').strip()))

# Pixel of screen
df_col=df.columns[0:]
for i in df_col:
    df[i]=df[i].astype('str')
    df[i]=df[i].apply(lambda x : x.strip())
        
def ex1_size(x):
    p=re.compile(r"\d\w{8}|\d\w{7}")
    return p.search(x)[0]
df['Pixel']=df['Screen'].apply(lambda x : ex1_size(x))

# screen mode 
df['full_hd']=df['Screen'].apply(lambda x : 1 if 'full hd' in x.lower() else 0)
df['ips panel']=df['Screen'].apply(lambda x : 1 if 'ips panel' in x.lower() else 0)
df['touchscreen']=df['Screen'].apply(lambda x : 1 if 'touchscreen' in x.lower() else 0)
df['4k ultra hd']=df['Screen'].apply(lambda x : 1 if '4k ultra hd' in x.lower() else 0)
df['quad hd+']=df['Screen'].apply(lambda x : 1 if 'quad hd+' in x.lower() else 0)
df['retina display']=df['Screen'].apply(lambda x : 1 if 'retina display' in x.lower() else 0)

# CPU
def ex2_size(x):
    p=re.compile(r"(^\w{5,}.\w{7}.\w{1,}.\w{1,})|(^\w{5}.\w{3,}.i\d{1})|(^\w{3,}.\w{4,}.[E]\d)|(^\w{3,}.\w{1,}.\w{6,})|(^\w{5,}.[A]\w{3})|(^\w{3,}.\w{2,}.\w)")
    return p.search(x)[0]
df['CPU-a']=df['CPU'].apply(lambda x : ex2_size(x))

def ex3_size(x):
    p=re.compile(r"(\d.\d\w{0,}z$)|(\d.\d\w{2,} $)|(\d\w{3}$)")
    return p.search(x)[0]
df['CPU-b']=df['CPU'].apply(lambda x : ex3_size(x))
df['CPU-b']=df['CPU-b'].apply(lambda x : x.split(' ')[1] if ' ' in x.lower() else x)

df['CPU-b']=df['CPU-b'].apply(lambda x : float(x.replace('GHz','').strip()))


# RAM
df['RAM']=df['RAM'].apply(lambda x : int(x.replace('GB','').strip()))

# Storage
df.rename(columns={' Storage':'Storage'},inplace=True)

df['Storage-1']=df['Storage'].apply(lambda x : x.split('+')[0].strip() if '+' in x.lower() else x)

df['Storage-2']=df['Storage'].apply(lambda x : x.split('+')[1].strip() if '+' in x.lower() else '')

def SSD_split(x):
    if 'ssd' in x['Storage-2'].lower() and 'ssd' in x['Storage-1'].lower():
        return x['Storage-1'].split(' ')[0] +' '+ x['Storage-2'].split(' ')[0]
    elif 'ssd' in x['Storage-1'].lower():
        return x['Storage-1'].split(' ')[0]
    elif 'ssd' in x['Storage-2'].lower():
        return x['Storage-2'].split(' ')[0]
    else : return 0
df['SSD']=df.apply(lambda x : SSD_split(x),axis=1)

def HDD_split(x):
    if 'hdd' in x['Storage-2'].lower() and 'hdd' in x['Storage-1'].lower():
        return x['Storage-1'].split(' ')[0] +' '+ x['Storage-2'].split(' ')[0]
    elif 'hdd' in x['Storage-1'].lower():
        return x['Storage-1'].split(' ')[0]
    elif 'hdd' in x['Storage-2'].lower():
        return x['Storage-2'].split(' ')[0]
    else : return 0
df['HDD']=df.apply(lambda x : HDD_split(x),axis=1)

df['Flash_storage']=df['Storage-1'].apply(lambda x : x.split(' ')[0] if 'flash storage' in x.lower() else 0)
df['Hybrid_storage']=df['Storage-1'].apply(lambda x : x.split(' ')[0] if 'hybrid' in x.lower() else 0)
df.drop(['Storage-1','Storage-2'],axis=1,inplace=True)

storage = df.columns[22:26]
for i in storage:
    df[i]=df[i].astype('str')
    df[i]=df[i].apply(lambda x : x.replace('TB','').replace('GB',''))
    
def storage_split(x):
    list = []
    if ' ' not in x.strip():
        c = int(x)
        return c
    elif ' ' in str(x.strip()):
        a = int(x.split(' ')[0])
        b = int(x.split(' ')[1])
        c = a+b
        return c
    list.append(c)
    return ' '.join(list)
df['SSD']=df['SSD'].apply(lambda x : storage_split(x))
df['HDD']=df['HDD'].apply(lambda x : storage_split(x))

for i in storage:
    df[i]=df[i].astype('int')
    df[i]=df[i].apply(lambda x : x*1024 if 0<x<4 else x)

# Operating System Version 

df['Operating System Version']=df['Operating System Version'].apply(lambda x : x.replace('nan','No_ver').replace('X','No_ver'))

# GPU

def ex4_size(x):
    p=re.compile(r"(^\w{1,}.\w{1,}.\w{7,}.)|(^\w{1,}.\w{1,}.\w{4,}.\w{7,})|(^\w{1,}.\w{7,})|(^\w{2,}.\w{6,})|(^\w{2,}.\w{1,}.\w{2,})") 
    return p.search(x)[0]
df['GPU-a']=df['GPU'].apply(lambda x : ex4_size(x))
df['GPU-a']=df['GPU-a'].apply(lambda x : x.replace('GTX1050','').replace('GTX 980','GeForce').strip())

def ex5_size(x):
    p=re.compile(r"(\d{1,}$)|(\d{3,}.\w{1,}$)|(\w{2,}$)|(\w{1,}...\w{1,}>$)")
    return p.search(x)[0]
df['GPU-b']=df['GPU'].apply(lambda x : ex5_size(x))

df['GPU-b']=df['GPU-b'].apply(lambda x : x.replace('<U+039C>','').replace('SLI','').replace('GTX','').strip())
df['MX_GPU']=df['GPU-b'].apply(lambda x : x.replace('MX','') if x.endswith('MX')or x.startswith('MX') else '' )
df['Nvidia_Ti-card']=df['GPU-b'].apply(lambda x : x.replace('Ti','').strip() if x.endswith('Ti') else '')
df['Quaro_card']=df['GPU-b'].apply(lambda x : x.replace('M','').strip() if x.endswith('M') and x.startswith('M') else '' )
df['firepro_card']=df['GPU-b'].apply(lambda x : x.replace('W','').replace('M','').strip() if x.endswith('M') and x.startswith('W') else '' )
df['AMD-MX_card']=df['GPU-b'].apply(lambda x : x.replace('M','').replace('X','').strip() if x.endswith('X') and x.startswith('M') else '' )
df['M-amd-qr']=df['GPU-b'].apply(lambda x : x.replace('M','').strip() if x.endswith('0') or x.endswith('5') and x.startswith('M') else 0)
df['Graphics_GPU']=df['GPU-b'].apply(lambda x : 530 if 'graphics' in x.lower().strip() else '' )

def fix_amd_qr(x):
    if x.startswith('MX'):
        return 0
    elif x.startswith('M') and x.endswith('0'):
        return x.replace('M','')
    elif x.startswith('M') and x.endswith('5'):
        return x.replace('M','')
    else : return 0

df['M-amd-qr']=df['GPU-b'].apply(lambda x : fix_amd_qr(x))
df['M-amd-qr']=df['M-amd-qr'].astype('int')

df['Quaro_card2']=df['M-amd-qr'].apply(lambda x : x if x>600 else '')
df['AMD_card']=df['M-amd-qr'].apply(lambda x : x if 0<x<600 else '')

ab_card = ['Quaro_card2','Quaro_card','AMD-MX_card','AMD_card','M-amd-qr']
for i in ab_card:
    df[i]=df[i].astype('str')

df['Quaro_GPU']=df.apply(lambda x : x['Quaro_card']+x['Quaro_card2'],axis=1)
df['AMD_Card']=df.apply(lambda x : x['AMD_card']+x['AMD-MX_card'],axis=1)
df.drop(ab_card,axis=1,inplace=True)

df['GPU-b']=df['GPU-b'].astype('str')
df['num_card']=df['GPU-b'].apply(lambda x : str(x) if x.isdigit() else 0)

gr_amd_c = ['520','620','500','400','505','530','70','630','510','430','405','510','515']
gr_amd_r= [ '555','5300','6000','580','615','455','650','540','560','550']
gtx = [ '1050','1060','1070','980','920','960','1080']
fix = ['5300','6000','4190','5130']
df['Gr_AMDc']=df['num_card'].apply(lambda x : x if x in gr_amd_c else '')
df['Gr_AMDr']=df['num_card'].apply(lambda x : x if x in gr_amd_r else '')
df['GTX_card']=df['num_card'].apply(lambda x : x if x in gtx else '')
df['Graphics_GPU']=df['Graphics_GPU'].astype('str')
df['Gr_AMDc']=df.apply(lambda x: x['Graphics_GPU']+x['Gr_AMDc']+x['AMD_Card'],axis=1)
df['Gr_AMDr']=df['Gr_AMDr'].apply(lambda x : x.replace('000','00').replace('5300','530').strip() if x.strip() in fix else x)
df['Gr_AMDr']=df.apply(lambda x : x['Gr_AMDr']+x['firepro_card']+x['Quaro_GPU'],axis=1)
df['GTX_card']=df.apply(lambda x : x['GTX_card']+x['Nvidia_Ti-card'],axis=1)
df['Gr_AMDr']=df['Gr_AMDr'].apply(lambda x : x.replace('4190','455').replace('5130','530').strip() if x.strip() in fix else x)
df.drop(['AMD_Card','num_card','Graphics_GPU','firepro_card','Nvidia_Ti-card','Quaro_GPU'],axis=1,inplace=True)

last_col=df.columns[28:]
for i in last_col:
    df[i]=df[i].apply(lambda x : x.strip())
    df[i]=df[i].apply(lambda x : x.strip() if x.isdigit() else 0)
for i in last_col:
    df[i]=df[i].astype('int')
    
# Weight 

df['Weight']=df['Weight'].apply(lambda x : float(x.replace('kg','').strip()))

df.to_csv('laptop_clean.csv')


#df_1=df.loc[df['GTX_card']=='1050']



















    

