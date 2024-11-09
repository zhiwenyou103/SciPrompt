import requests
import urllib
import json


def fetchRelatedWords(word):
    '''
    using https://relatedwords.org/ find the related words to the given word
    returns list of dict, in descending order of relatedness score
    for eg:
        {'word': 'health', 'score': 27.688784532941305, 'from': 'cn5,ol,wiki,swiki'}
        {'word': 'medicine', 'score': 24.990184188413064, 'from': 'cn5,ol,w2v,wiki,swiki'}
        {'word': 'treatment', 'score': 8.791995623154092, 'from': 'cn5,cn5,w2v,swiki'}
        ...
    '''
    url = "https://relatedwords.org/api/related?term=" + urllib.parse.quote(word)
    try:
        if word == 'open-domain question answering':
            another_url = "https://reversedictionary.org/api/related?term=" + urllib.parse.quote(word)
            page = requests.get(another_url)
            related_words = json.loads(page.content)
            
            filtered_words = [word_data for word_data in related_words if word_data['score'] > 0]
            return filtered_words
        page = requests.get(url)
        related_words = json.loads(page.content)
        filtered_words = [word_data for word_data in related_words if word_data['score'] > 0]
        if not filtered_words:
            another_url = "https://reversedictionary.org/api/related?term=" + urllib.parse.quote(word)
            page = requests.get(another_url)
            related_words = json.loads(page.content)
            
            filtered_words = [word_data for word_data in related_words if word_data['score'] > 0]
        
        return filtered_words
    except Exception as e:
        print("-"*40, "ERROR:\n", e, "\n", "-"*40)
        return []