Run: 
./download.sh
python3 -m squad.prepro
python3 -m bidaf.cli --mode train

To debug:
python3 -m bidaf.cli --mode train --debug True
