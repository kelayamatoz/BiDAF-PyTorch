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

TODOs:
- Saving model weights in main.py
- Multi-GPU training support in model.py and trainer.py
- Summary at the right epoch
- Loss function
- embedded matrix
- Update requirements.txt
