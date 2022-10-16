import requests
import re
from bs4 import BeautifulSoup
import html5lib

def url_to_html(url):
    """Scrapes the html content from a web page. Takes a URL string as input and returns an html object. """
    
    # Get the html content
    res = requests.get(url, headers={"User-Agent": "XY"})
#     res = requests.get(url + ".pdf", headers={"User-Agent": "XY"})
    html = res.text
    parser_content = BeautifulSoup(html, 'html5lib')
    return parser_content

# We are looking for the author information at places where it can often be found.
# If we do not find it, it does not mean that it is not there.
def parse_author(html_content):
    
    # Initialize variables
    search_query = re.compile('author', re.IGNORECASE)
    name = ""
    
    # The author information might be encoded as a value of the attribute name
    attribute = html_content.find('meta', attrs={'name': search_query})
    
    # Or as a property
    property = html_content.find('meta', property=search_query)

    found_author = attribute or property
    
    if found_author:
        name = found_author['content']
   
   # If the author name cannot be found in the metadata, we might find it as an attribute of the text.
    else:
        itemprop = html_content.find(attrs={'itemprop': 'author'})
        byline = html_content.find(attrs={'class': 'byline'})
    
        found_author = itemprop or byline
        
        if found_author:
            name = found_author.text
    
    name = name.replace("by ", "")
    name = name.replace("\n", "")
    return name.strip()


#This function requires the HTML content of the result as an input parameter
#It returns the actual text content

def parse_news_text(html_content):

    # Try to find Article Body by Semantic Tag
    article = html_content.find('article')

    # Otherwise, try to find Article Body by Class Name (with the largest number of paragraphs)
    if not article:
        articles = html_content.find_all(class_=re.compile('(body|article|main)', re.IGNORECASE))
        if articles:
            article = sorted(articles, key=lambda x: len(x.find_all('p')), reverse=True)[0]
            

    # Parse text from all Paragraphs
    text = []
    if article:
        for paragraph in [tag.text for tag in article.find_all('p')]:
            if re.findall("[.,!?]", paragraph):
                text.append(paragraph)
    text = re.sub(r"\s+", " ", " ".join(text))

    return text

def parse_news_text_revision(html_content):

    # Try to find Article Body by Semantic Tag
    article = html_content.find('article')

    # Otherwise, try to find Article Body by Class Name (with the largest number of paragraphs)
    if not article:
        articles = html_content.find_all(class_=re.compile('(body|article|main)', re.IGNORECASE))
        if articles:
            article = sorted(articles, key=lambda x: len(x.find_all('p')), reverse=True)[0]
            

    # Parse text from all Paragraphs
    text = []
    if article:
        for paragraph in [tag.text for tag in article.find_all('p')]:
            if re.findall("[。，！？]", paragraph):
                text.append(paragraph)
    text = re.sub(r"\s+", " ", " ".join(text))

    return text

def web_scrapping(keyword,language):
    
    if language == 'zh':
        url = "https://www.thenewslens.com/search/" + keyword
        url_page2 = "https://www.thenewslens.com/search/" + keyword+'?page=2'
        url_page3 = "https://www.thenewslens.com/search/" + keyword+'?page=3'
        url_page4 = "https://www.thenewslens.com/search/" + keyword+'?page=4'
        url_page5 = "https://www.thenewslens.com/search/" + keyword+'?page=5'
        url_page6 = "https://www.thenewslens.com/search/" + keyword+'?page=6'
    else:
        url = "https://www.techradar.com/search?searchTerm="+keyword
        url_page2 = "https://www.techradar.com/search/page/2?searchTerm="+keyword
        url_page3 = "https://www.techradar.com/search/page/3?searchTerm="+keyword
        url_page4 = "https://www.techradar.com/search/page/4?searchTerm="+keyword
        url_page5 = "https://www.techradar.com/search/page/5?searchTerm="+keyword
        url_page6 = "https://www.techradar.com/search/page/6?searchTerm="+keyword

    url_list = [url, url_page2, url_page3,url_page4,url_page5,url_page6]
    for urls in url_list:      
        print('The search request URL:', urls)
        
    parser_content = url_to_html(url) 
    parser_content2 = url_to_html(url_page2) 
    parser_content3 = url_to_html(url_page3) 
    parser_content4 = url_to_html(url_page4) 
    parser_content5 = url_to_html(url_page5) 
    parser_content6 = url_to_html(url_page6) 
        
    url_parser = [parser_content,parser_content2,parser_content3,parser_content4,parser_content5,parser_content6]
    
    outfile = "data/ai_"+language+"_overview.tsv"
    
    with open(outfile, "w",encoding="utf-8") as f:

        f.write("Publication Date\tTime\tAuthor\tTitle\tURL\tText\n")
        if language =='zh':
            div_list = []
            for i in url_parser:
                div_list.append(i.select("h2 a.gtm-track"))

            for i in range(6):

                for index, link in enumerate(div_list[i]): 
                    if i == 5:
                        if index == 10:
                            break

                    found_url = link["href"]
                    title = link['title']
                    parser_content= url_to_html(found_url)
                    attributes = parser_content.find_all('meta', attrs={'name': 'pubdate'}) 
                    author = parse_author(parser_content)
                    content = parse_news_text_revision(parser_content)
                    for attr in attributes:
                        datetime = attr['content']
                        date, time = datetime.split("T") 

                        content = content.replace("\n", "")
                        output = "\t".join([date, time, author, title, found_url, content])  
                        f.write(output +"\n")
        else:
            
            div_list = []
            for i in url_parser:
                div_list.append(i.select("div a.article-link"))

            for i in range(6):

                for index, link in enumerate(div_list[i]): 
                    if i == 5:
                        if index == 6:
                            break

                    found_url = link["href"]
                    title = link['aria-label']
                    parser_content= url_to_html(found_url)
                    attributes = parser_content.find_all('meta', attrs={'name': 'pub_date'}) 
                    author = parse_author(parser_content)
                    content = parse_news_text(parser_content)
                    for attr in attributes:
                        datetime = attr['content']
                        date, time = datetime.split("T") 


                        content = content.replace("\n", "")
                        output = "\t".join([date, time, author, title, found_url, content])  
                        f.write(output +"\n")
                        
keyword_en = 'artificial+intellgience' 
keyword_zh = '人工智慧'  
language_en = 'en'
language_zh = 'zh'

web_scrapping(keyword_en,'en')
web_scrapping(keyword_zh,'zh')
