To preprocess data: 
./download.sh
python3 -m squad.prepro

To run training:
python3 -m bidaf.cli --mode train

To debug:
python3 -m bidaf.cli --mode train --debug

directory structure:
/my: util files
/bidaf: bidaf training / testing files
/squad: squad dataset preprocessing files
