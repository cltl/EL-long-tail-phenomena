import redis
import urllib.parse
import pickle

rds=redis.Redis()

def computePR(url):
        val=rds.get('pr:%s' % url)
        return float(val) if val else 0.0

def normalizeURL(s):
        if s:
                return urllib.parse.unquote(s.replace("http://en.wikipedia.org/wiki/", "").replace("http://dbpedia.org/resource/", ""). replace("http://dbpedia.org/page/", "").strip().strip('"'))
        else:
                return '--NME--'

def getLinkRedirect(link):
        red=rds.get('rdr:%s' % link)
        if red:
                return red.decode('UTF-8')
        else:
                return link

def store_dataset(title, articles):    
    with open('bin/%s.bin' % title, 'wb') as outfile:
        pickle.dump(articles, outfile)
        
def store_system_data(dataset, system, articles):    
    with open('bin/%s_%s.bin' % (dataset, system), 'wb') as outfile:
        pickle.dump(articles, outfile)
