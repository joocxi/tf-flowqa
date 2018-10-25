# Download CoQA
DATA_DIR=data
mkdir -p $DATA_DIR
wget https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json -O $DATA_DIR/coqa-train-v1.0.json
wget https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json -O $DATA_DIR/coqa-dev-v1.0.json

# DownloadGlove

