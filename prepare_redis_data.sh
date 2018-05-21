mkdir to_cache
cd to_cache

wget http://downloads.dbpedia.org/2016-04/core-i18n/en/disambiguations_en.ttl.bz2

wget http://downloads.dbpedia.org/2016-04/core-i18n/en/redirects_en.ttl.bz2

wget http://people.aifb.kit.edu/ath/download/pagerank_en_2016-04.tsv.bz2


echo "Downloads done. Now extracting the archives to disk..."
bunzip2 *

echo "Now caching data to Redis..."

cd ..

python3 cache_data.py

echo "Caching done. Now removing the files..."
rm -r to_cache
