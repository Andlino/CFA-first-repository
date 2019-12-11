#%%
### CAMELOT FOR TABLE EXTRACTION
# %%

#pip install pdfminer

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter#process_pdf
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO

def pdf_to_text(pdfname):

    # PDFMiner boilerplate
    rsrcmgr = PDFResourceManager()
    sio = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, sio, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    # Extract text
    fp = open(pdfname, 'rb')
    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
    fp.close()

    # Get text from StringIO
    text = sio.getvalue()

    # Cleanup
    device.close()
    sio.close()

    return text

# %%
pdfname = "C:/Users/au615270/Documents/Python/coalition_governments_and_party_competition_political_communication_strategies_of_coalition_parties.pdf"

parser = pdf_to_text(pdfname)
fulltext = parser.splitlines()


#%%
#testlist = []

#for line in fulltext:
#    if line != '': 
#        a = (line)
#    if line == '': 
#        testlist.append(a)    


# %%

import camelot
tables = camelot.read_pdf('C:/Users/au615270/Documents/Python/coalition_governments_and_party_competition_political_communication_strategies_of_coalition_parties.pdf')
tables
#TableList n=1
tables.export('foo.csv', f='csv', compress=True) # json, excel, html
tables[0]
#Table shape=(7, 7)
tables[0].parsing_report
{
    'accuracy': 99.02,
    'whitespace': 12.24,
    'order': 1,
    'page': 1
}

# %%
