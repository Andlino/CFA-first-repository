#%% [markdown]
# XML-Extractor regular expression version 1
# September 2019

#%%
import requests
from xml.dom import minidom
import xml.etree.ElementTree as ET
import csv
from csv import reader
import pandas as pd
from bs4 import BeautifulSoup
import requests

# <ID>_struct.txt sect_ID sect_HEADER para_ID para_text
# testing section ID & header

#%%

sect_df = pd.DataFrame(columns=['ID', 'Title'])


for child in root.iter('*'):
        for attrib in child: 
                if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                        sect_df = sect_df.append({'ID': [str(attrib.get('id'))], 'Title': [str(attrib.text)]}, ignore_index=True)


para_df = pd.DataFrame(columns=['ID', 'Paragraph'])

for child in root.iter('*'):
    for attrib in child: 
        if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}para' and '{http://www.elsevier.com/xml/common/dtd}cross-refs'):
            para_df = para_df.append({'ID': [str(attrib.get('id'))], 'Paragraph' : [str(attrib.text)]}, ignore_index=True)

print(para_df)

#%%

# <ID>_reflink.txt para_id ref_id



for child in root.iter('*'):
        for attrib in child: 
                if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}cross-ref'):
                    print(attrib.get('id'))
                    print(attrib.get('refid'))
                    print(attrib.text)
                    print("---------------------")


#%%

#%%

isIntroduction = False

for child in root.iter('*'):
        for attrib in child:
                if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                        if attrib.text == 'Introduction':
                                isIntroduction = True
                if isIntroduction == True:
                        if attrib.tag == '{http://www.elsevier.com/xml/common/dtd}para':
                                print(attrib.text)
                if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                        if attrib.text != 'Introduction':
                                isIntroduction = False




#%%

isIntroduction = False

for child in root.iter('*'):
        for attrib in child:
                if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                        if attrib.text == 'Simulations and results':
                                isIntroduction = True
                if isIntroduction == True:
                        if attrib.tag == '{http://www.elsevier.com/xml/common/dtd}para':
                                print(attrib.text)
                if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                        if attrib.text != 'Simulations and results':
                                isIntroduction = False


#%%

#new trail with working iteration 
#<ID>_struct.txt : sect_id      sect_header     para_id

#sect_id

sect_df = pd.DataFrame(columns=['ID', 'Title'])


for child in root.iter('*'):
        for attrib in child: 
                if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                        sect_df = sect_df.append({'ID': [str(attrib.get('id'))], 'Title': [str(attrib.text)]}, ignore_index=True)


#%%

import pandas as pd
import re

section_list = sect_df['Title'].tolist()

# You need to flat your list of lists, idiot!
flat_list = [item for sublist in section_list for item in sublist]


#section_list = [str(a) for a in section_list]
#section_list = ", " . join(section_list)
#section_list = re.sub("\[", "", section_list)
#section_list = re.sub("\]", "", section_list)

#for  i in range(len(section_list)) :
#    print(section_list[i]) 
#   i += 1

#%%

isIntroduction = False

for child in root.iter('*'):
        for attrib in child:
                for i in range(len(flat_list)):
                        if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                                if attrib.text == flat_list[i]:
                                        isIntroduction = True
                        if isIntroduction == True:
                                if attrib.tag == '{http://www.elsevier.com/xml/common/dtd}para':
                                        print(attrib.text)
                                        print(attrib.get('id'))
                        if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                                if attrib.text != flat_list[i]:
                                        isIntroduction = False




#%%

isIntroduction = False


for i in range(len(flat_list)):
        for child in root.iter('*'):
                for attrib in child:
                        if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                                if attrib.text == flat_list[i]:
                                        print("*[(" + attrib.text + ")]*")
                                        isIntroduction = True
                        if isIntroduction == True:
                                if attrib.tag == '{http://www.elsevier.com/xml/common/dtd}para':
                                        print(attrib.text)
                                        print(attrib.get('id'))
                                        print('--------------------')
                        if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                                if attrib.text != flat_list[i]:
                                        isIntroduction = False
#%%

isIntroduction = False
isPara = False

for child in root.iter('*'):
        for attrib in child:
                if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                        if attrib.text == 'Introduction':
                                print(attrib.get('id'))
                                isIntroduction = True
                if isIntroduction == True:
                        if attrib.tag == '{http://www.elsevier.com/xml/common/dtd}para':
                                isPara = True
                                print(attrib.text)
                if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                        if attrib.text != 'Introduction':
                                isIntroduction = False
                if isPara == True: 
                        if attrib.tag == '{http://www.elsevier.com/xml/common/dtd}cross-refs':
                                print(attrib.text)




#%%


#new trail with working iteration 
#<ID>_struct.txt : sect_id      sect_header     para_id        para_content

#sect_id

sect_id_df = pd.DataFrame(columns=['ID'])

sect_header_df = pd.DataFrame(columns=['Title'])

para_id_df = pd.DataFrame(columns=['id'])

para_content_df = pd.DataFrame(columns=['content'])

joint_df1 = pd.DataFrame(columns=['ID', 'Title'])
joint_df2 = pd.DataFrame(columns=['id', 'content'])

isa = False
isb = False
isc = False

for i in range(len(flat_list)):
        for child in root.iter('*'):
                for attrib in child:
                        if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                                if attrib.text == flat_list[i]:
                                        joint_df1 = joint_df1.append({'ID' : [str(attrib.get('id'))], 'Title' : [str(attrib.text)]}, ignore_index=True)
                                        isa = True
                        if isa == True:
                                if attrib.tag == '{http://www.elsevier.com/xml/common/dtd}para':
                                        joint_df2 = joint_df2.append({'id' : [str(attrib.get('id'))], 'content' : [str(attrib.text)]}, ignore_index=True)
                        if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                                if attrib.text != flat_list[i]:
                                        isa = False

test = pd.concat([joint_df1, joint_df2], axis=1)
#joint_df.ffill(axis= 0)
#%%

isa = False
isb = False
isc = False


joint_df1 = pd.DataFrame(columns=['ID', 'Title'])
joint_df2 = pd.DataFrame(columns=['id', 'content'])
joint_new = pd.DataFrame(columns=['ID', 'Title'])

for i in flat_list:
        for child in root.iter('*'):
                for attrib in child:
                        if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                                if attrib.text == i:
                                        joint_df1 = joint_df1.append({'ID' : [str(attrib.get('id'))], 'Title' : [str(attrib.text)]}, ignore_index=True)
                                        isa = True
                        if isa == True:
                                if attrib.tag == '{http://www.elsevier.com/xml/common/dtd}para':
                                        joint_df2 = joint_df2.append({'id' : [str(attrib.get('id'))], 'content' : [str(attrib.text)]}, ignore_index=True)
                        if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                                if attrib.text != i:
                                        isa = False
        for d in range(len(joint_df2['id'])):
                joint_new = joint_new.append({'ID' : [joint_df1['ID'].to_string(index=False)], 'Title' : [joint_df1['Title'].to_string(index=False)]}, ignore_index=True)
                testjoint = pd.concat([joint_new, joint_df2], axis=1)


#for i in range(len(joint_df2['id'])):
#        joint_new = joint_new.append({'ID' : [joint_df1['ID'].to_string(index=False)], 'Title' : [joint_df1['Title'].to_string(index=False)]}, ignore_index=True)
#        testjoint = pd.concat([joint_new, joint_df2], axis=1)


#%%


isa = False
isb = False
isc = False


joint_df1 = pd.DataFrame(columns=['ID', 'Title'])
joint_df2 = pd.DataFrame(columns=['id', 'content'])
joint_new = pd.DataFrame(columns=['ID', 'Title'])

for child in root.iter('*'):
        for attrib in child:
                if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                        if attrib.text == 'Introduction':
                                joint_df1 = joint_df1.append({'ID' : [str(attrib.get('id'))], 'Title' : [str(attrib.text)]}, ignore_index=True)
                                isa = True
                if isa == True:
                        if attrib.tag == '{http://www.elsevier.com/xml/common/dtd}para':
                                joint_df2 = joint_df2.append({'id' : [str(attrib.get('id'))], 'content' : [str(attrib.text)]}, ignore_index=True)
                if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                        if attrib.text != 'Introduction':
                                isa = False
for d in joint_df2['id']:
        joint_new = joint_new.append({'ID' : [joint_df1['ID'].to_string(index=False)], 'Title' : [joint_df1['Title'].to_string(index=False)]}, ignore_index=True)
        testjoint = pd.concat([joint_new, joint_df2], axis=1)



#testframe = pd.DataFrame(columns=['test'])

#for i in range(len(para_id_df)):
#        testframe = testframe.append({'test' : sect_header_df['Title'].to_string(index=False)}, ignore_index=True)


#%%


isa = False
isb = False
isc = False

df = pd.DataFrame({'ID': [], 'Title': [], 'id': [], 'content' : []})

for i in flat_list:
        for child in root.iter('*'):
                for attrib in child:
                        if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                                if attrib.text == i:
                                        df = df.append({'ID' : [str(attrib.get('id'))], 'Title' : [str(attrib.text)]}, ignore_index=True)
                                        isa = True
                        if isa == True:
                                if attrib.tag == '{http://www.elsevier.com/xml/common/dtd}para':
                                        df = df.append({'id' : [str(attrib.get('id'))], 'content' : [str(attrib.text)]}, ignore_index=True)
                        if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                                if attrib.text != i:
                                        isa = False

#df.to_csv('testfilenew.csv', index = None, header=True)

#%%

# <ID>_reflink.txt : para_id            ref_id


isa = False
isb = False

df2 = pd.DataFrame({'paraID': [], 'refID': []})

for i in flat_list:
        for child in root.iter('*'):
                for attrib in child:
                        if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                                if attrib.text == i:
                                        isa = True
                        if isa == True:
                                if attrib.tag == '{http://www.elsevier.com/xml/common/dtd}para':
                                        print(attrib.get('id'))
                                        df2 = df2.append({'paraID' : [str(attrib.get('id'))]}, ignore_index=True)
                                        isb = True
                        if isb == True: 
                                if attrib.tag == '{http://www.elsevier.com/xml/common/dtd}cross-ref':
                                        print(attrib.get('refid'))
                                        print(attrib.get('id'))
                                        print('-----------------')
                                        df2 = df2.append({'refID' : [str(attrib.get('refid'))]}, ignore_index=True)
                        if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                                if attrib.text != i:
                                        isa = False

#%%

para_df = pd.DataFrame(columns=['ID'])


for child in root.iter('*'):
        for attrib in child: 
                if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}para'):
                        para_df = para_df.append({'ID': [str(attrib.get('id'))]}, ignore_index=True)



para_list = para_df['ID'].tolist()

flat_list2 = [item for sublist in para_list for item in sublist]

#%%

for t in flat_list2:
        for child in root.iter('*'):
                for attrib in child: 
                        if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}para'):
                                if attrib.get('id') == t:
                                        print(attrib.text)
                                        print(attrib.get('id'))
                                        isa = True
                        if isa == True:
                                if attrib.tag == '{http://www.elsevier.com/xml/common/dtd}cross-refs':
                                        print(attrib.get('id'))
                                        print(attrib.get('refid'))
                                if attrib.tag == '{http://www.elsevier.com/xml/common/dtd}cross-ref':
                                        print(attrib.get('id'))
                                        print(attrib.get('refid'))
                                        print('--------------------------')
                        if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}para'):
                                if attrib.get('id') != t:
                                        isa = False




#%%


for child in root.iter('*'):
        for attrib in child: 
                if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                        if attrib.text == 'Introduction':
                                for elem in attrib.iter('*'):
                                        print(elem.text)
                                        print(elem.tag)
                                        print('--------------------')
                                        if (elem.tag=='{http://www.elsevier.com/xml/common/dtd}cross-refs'):
                                                print(elem.text)
#%%

isa = False

for child in root.iter('*'):
        for attrib in child:
                if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}para'):
                        if attrib.get('id') == 'p0010':
                                print(attrib.text)
                                isa = True
                if isa == True:
                        if attrib.tag == '{http://www.elsevier.com/xml/common/dtd}cross-refs':
                                print(attrib.get('id'))
                                print(attrib.get('refid'))
                if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}section-title'):
                        if attrib.get('id') == 'p0010':
                                isa = False

#%%

for child in root.iter('*'):
        for attrib in child:
                if (attrib.tag=='{http://www.elsevier.com/xml/common/dtd}para'):
                        if attrib.get('id') == 'p0010':
                                print(list(attrib))

#%%
