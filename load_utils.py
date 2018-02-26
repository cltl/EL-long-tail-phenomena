import redis
import urllib.parse
import pickle
import json
import urllib.request
from urllib.request import urlopen, Request
from urllib.parse import urlencode

rds=redis.Redis()

agdistis_url = "http://akswnc9.informatik.uni-leipzig.de:8113/AGDISTIS"

def computePR(url):
    """
    Retrieve a pagerank value for a URI from Redis.
    """
    val=rds.get('pr:%s' % url)
    return float(val) if val else 0.0

def normalizeURL(s):
    """
    Normalize a URI by removing its Wikipedia/DBpedia prefix.
    """
    if s:
        return urllib.parse.unquote(s.replace("http://en.wikipedia.org/wiki/", "").replace("http://dbpedia.org/resource/", ""). replace("http://dbpedia.org/page/", "").strip().strip('"'))
    else:
        return '--NME--'

def getLinkRedirect(link):
    """
    If a link is a redirect, get its target entity.
    """
    red=rds.get('rdr:%s' % link)
    if red:
        return red.decode('UTF-8')
    else:
        return link

def store_dataset(title, articles):  
    """
    Store a dataset object to a pickle file.
    """  
    with open('bin/%s.bin' % title, 'wb') as outfile:
        pickle.dump(articles, outfile)

def disambiguate_agdistis(xml_text, type='agdistis'):
	"""
    Send text to AGDISTIS/MAG to disambiguate, and return the answer JSON.
    """
	params={"text": xml_text, "type": type}
	request = Request(agdistis_url, urlencode(params).encode())
	this_json = urlopen(request).read().decode()
	return json.loads(this_json)
        
def store_system_data(dataset, system, articles):
    """
    Store an object containing system-processed data to a pickle file.
    """  
    with open('bin/%s_%s.bin' % (dataset, system), 'wb') as outfile:
        pickle.dump(articles, outfile)
