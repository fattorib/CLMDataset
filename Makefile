raw:
	gdown https://drive.google.com/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx

unzip:
	tar -xvf  'openwebtext.tar.xz' -C 'data/raw/train'

untarall:
	for f in *.xz; do tar xf "$f"; done

download: 
	wget https://zenodo.org/record/3834942/files/openwebtext.tar.xz

	
for f in */*.txt; do echo cp -t txtonly "$f"; done