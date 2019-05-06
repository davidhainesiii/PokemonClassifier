import requests
import re
import pandas as pd


from azure.cognitiveservices.search.imagesearch import ImageSearchAPI
from msrest.authentication import CognitiveServicesCredentials

subscription_key = "7276008dbec5468aa270174dc73cd8c2"
client = ImageSearchAPI(CognitiveServicesCredentials(subscription_key))

data = pd.read_csv('pokemon_aug.csv')


def pokeframer(search_terms):
    tempframe = pd.DataFrame()
    for search_term in search_terms:
        image_results = client.images.search(query=search_term)
        for i in range(0, len(image_results.value)):
            url = image_results.value[i].content_url
            ext = re.findall(r'\.[a-z]+$', url)
            try:
                r = requests.get(image_results.value[i].content_url)
            except requests.exceptions.SSLError:
                pass
            except requests.exceptions.ConnectionError:
                pass
            try:
                fname = str(search_term+"-"+str(i)+ext[0])
            except IndexError:
                pass
            except ConnectionError:
                pass
            entry = {"Type1":data.Type1.loc[data.Name == search_term], "fileid":fname}
            eframe = pd.DataFrame(entry, index=[0])
            tempframe.append(eframe, ignore_index=True)
            print('Appending file:'+fname)
    return tempframe


newframe = pokeframer(data.Name)
newframe.to_csv('pokemon_bigimage_framed.csv')
