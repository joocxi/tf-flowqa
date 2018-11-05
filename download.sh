# Download CoQA
DATA_DIR=data
mkdir -p $DATA_DIR

# download CoQA dataset
wget https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json -O $DATA_DIR/coqa-train-v1.0.json
wget https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json -O $DATA_DIR/coqa-dev-v1.0.json

# download QuAC dataset
wget https://s3.amazonaws.com/my89public/quac/train_v0.2.json -O $DATA_DIR/quac_train_v0.2.json
wget https://s3.amazonaws.com/my89public/quac/val_v0.2.json -O $DATA_DIR/quac_val_v0.2.json

# DownloadGlove
