unzip:
	tar -xvf  'openwebtext.tar.xz' -C 'data/raw/train'

untarall:
	for f in *.xz; do tar xvf "$f"; done

download: 
	wget https://zenodo.org/record/3834942/files/openwebtext.tar.xz

	
for f in */*.txt; do cp -t txtonly "$f"; done