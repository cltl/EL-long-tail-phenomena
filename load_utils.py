import redis
import urllib.parse

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

