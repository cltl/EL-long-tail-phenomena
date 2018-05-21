cd data
if [ ! -d "$DIRECTORY" ]; then
    mkdir N3
fi
cd N3
wget https://raw.githubusercontent.com/dice-group/n3-collection/master/RSS-500.ttl
wget https://raw.githubusercontent.com/dice-group/n3-collection/master/Reuters-128.ttl
cd ../..
