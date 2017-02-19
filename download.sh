mkdir -p cat
mkdir -p human

wget -O CatOpen.zip https://sites.google.com/site/catdatacollection/data/CatOpen.zip?attredirects=0&d=1
unzip CatOpen.zip
mkdir CatOpen/test
mkdir CatOpen/train
mv CatOpen/CAT_03_Open CatOpen/test/
mv CatOpen/CAT_00_Open CatOpen/train/
mv CatOpen/CAT_01_Open CatOpen/train/
mv CatOpen/CAT_02_Open CatOpen/train/
mv CatOpen/CAT_04_Open CatOpen/train/
mv CatOpen/CAT_05_Open CatOpen/train/
mv CatOpen/CAT_06_Open CatOpen/train/