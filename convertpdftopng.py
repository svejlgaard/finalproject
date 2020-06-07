# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 14:58:35 2020

@author: Sapientia
"""

import os
from pdf2image import convert_from_path

pdf_dir = r"D:\Sapientia\Dropbox\Fysik på KU\Big Data Analysis\Final Project\plots"
os.chdir(pdf_dir)

for pdf_file in os.listdir(pdf_dir):

        if pdf_file.endswith(".pdf"):

            pages = convert_from_path(pdf_file, 600)
            pdf_file = pdf_file[:-4]

            for page in pages:

               page.save("%s-page%d.png" % (pdf_file,pages.index(page)), "PNG")