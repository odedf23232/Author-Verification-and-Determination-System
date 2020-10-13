from io import StringIO

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage


def convert_pdf_to_txt(path: str) -> str:
    """
    Loads a PDF file, and converts it to raw text.

    :param path: path of PDF file
    :return: raw text in the PDF file
    """
    # copied from
    # https://stackoverflow.com/questions/26748788/extraction-of-text-from-pdf-with-pdfminer-gives-multiple-copies
    # modified to fix potential resource leaking

    rsrcmgr = PDFResourceManager()
    codec = 'utf-8'
    laparams = LAParams()

    with StringIO() as retstr, \
            TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams) as device, \
            open(path, 'rb') as fp:
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()

        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                                      check_extractable=True):
            interpreter.process_page(page)

        text = retstr.getvalue()

    return text
