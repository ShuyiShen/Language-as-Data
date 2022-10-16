# Langauge as Data - final Project: Data Analysis & Blog Post
Group Assignment for the Course Langauge as Data at VU University Amsterdam

This dataset has been created for the purpose of setting up a linguistic comparative analysis between the languages 
'Chinese' and 'English', specified on the controversial topic 'Artificial Intelligence.'

## AUTHORS
------------------
S.Shen(Shuyi Shen), Sharona Badloe

### Research question: 
------------------
- What are the differences in terminology use between English and Chinese in our dataset
- how do these differences influence the framing of the topic?

## PROJECT STRUCTURE
-------------------
**What you will find in this project:**

The folder contains the files as followed:
- a file get_all_documents.py that extracts the 100 dataset for two languages 
- a file evaluate_annotation.py that outputs the inter-annotator agreement
- a file run_all_analyses.py that runs all analyses and outputs results and plots.
- annotation_guidelines.pdf: a document providing our annotation categories and detailed annotation guidelines with examples.
- basic_statistics.pdf: 
    - a document presenting tables that contain basic statistics of the metadata, titles in words and in characters, and the frequency of author names that occur in the dataset.
- analysis.pdf: A text document containing 
  - a). a description and evaluation of the annotation task
  - b). the word-based analysis. 
- annotationsheet_term_annotator.csv: 
  - 6 annotation sheets for two languages with annotations from two annotators for three terms
- chinese_stopword.txt
  - a txt file that contains chinese stopwords 
- SourceHanSansTW-Regular.otf
  - a file that contains chinese font for world cloud visualization

------------------------------------------------------------------------------------------------------------------------
**Notice:**

Considering we specify the relative path to our inputfile and outputfile in most of our functions as
 "../data/.." across all the py files, we stored 6 annotation sheets and 2 tsv files for two languages 
in a separate folder, called "data". 

To remove the chinese stopwords and visualize the wordcloud for chinese, we also save 'chinese_stopword.txt' 
and 'SourceHanSansTW-Regular.otf' to the 'data' folder. 

Before we run the py files, please download the 'wiki.zh.vec'(chinese) and 'wiki-news-300d-1M.vec' 
from "https://fasttext.cc/docs/en/pretrained-vectors.html" and add them to the 'data' folder. 

------------------------------------------------------------------------------------------------------------------------
## License information

License information

- All TechRadar data belongs to Future plc. Content may be printed or downloaded for personal use only, provided that each copy contains a 
notice that the Material is owned by or licensed to Future or its group companies.
https://www.futureplc.com/terms-conditions/

- All The News Lens Data belongs to TNL Media Group. The content of this service provided by the Group and any information on the website 
that can be viewed, listened to, or obtained through other methods are for personal and non-commercial use only.
https://www.tnlmedia.com/terms

We scraped data from two sources:

- The News Lens: Chinese tech-news website
https://www.thenewslens.com/

- TechRadar: English (UK) tech-news website 
https://www.techradar.com/