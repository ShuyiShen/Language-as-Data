import pandas as pd
import glob
import os.path
import random
from itertools import combinations
from sklearn.metrics import cohen_kappa_score, confusion_matrix

terms = ["privacy", "data", "security", "隱私", "數據", "安全"]
categories = ["positive", "negative", "neutral"]

# When we use the random function, we should set a fixed seed, if our results should be reproducible
random.seed(733)

for term in terms:
    annotations = {}

    # Read in the data
    for sheet in glob.glob("data/annotationsheet_" + term +"*.tsv"):
        filename, extension = os.path.basename(sheet).split(".")
        prefix, term, annotator = filename.split("_")

        # Read in annotations
        annotation_data = pd.read_csv(sheet, sep="\t", header=0, keep_default_na=False)
        annotations[annotator] = annotation_data["Annotation"]

        # The example sheets are not annotated. I am using random annotations here. Make sure to comment this part out.
        annotations[annotator] = random.choices(categories, k=len(annotation_data))

    annotators = annotations.keys()
    print(annotators)
   

    for annotator_a, annotator_b in combinations(annotators, 2):
        agreement = [anno1 == anno2 for anno1, anno2 in  zip(annotations[annotator_a], annotations[annotator_b])] 
        percentage = sum(agreement)/len(agreement)
        print(annotator_a, annotator_b)
        print("Percentage Agreement: %.2f" %percentage)
        kappa = cohen_kappa_score(annotations[annotator_a], annotations[annotator_b], labels=categories)
        print("Cohen's Kappa: %.2f" %kappa)
        confusions = confusion_matrix(annotations[annotator_a], annotations[annotator_b], labels=categories)
        print(confusions)
        print


# Different ways to output a table
confusionmatrix1 = {"positive": [0,2,3], "negative": [ 2,0,3], "neutral": [3,3,4]}
confusionmatrix2 = {"positive": [5,2,3], "negative": [ 1,2,2], "neutral": [2,0,3]}
confusionmatrix3 = {"positive": [4,2,0], "negative": [ 4,1,3], "neutral": [2,1,3]}
confusionmatrix4 = {"positive": [2,1,2], "negative": [ 1,1,4], "neutral": [3,4,2]}
confusionmatrix5 = {"positive": [4,3,2], "negative": [ 3,3,1], "neutral": [2,2,0]}
confusionmatrix6 = {"positive": [3,2,3], "negative": [ 0,4,3], "neutral": [4,0,1]}

for i in [confusionmatrix1,confusionmatrix2,confusionmatrix3,confusionmatrix4,
          confusionmatrix5,confusionmatrix6]:

    pandas_table = pd.DataFrame(i, index=["positive", "negative", "neutral"])
    print(pandas_table)
    markdown_table = pandas_table.to_markdown()
    print(markdown_table)
    print()
    latex_table = pandas_table.to_latex(caption="Confusion matrix for annotator a1 and a2 for the term \emph{meat}")
    print(latex_table)
