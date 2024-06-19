import argparse

from data_preprocessing import TwitterDataProcessing
from models import lda_model, bertopic_model


if __name__ == "__main__":
    """Orchestrates topic modeling: data preprocessing and then model traning. 
    Prints the top 10 topics
    """

    parser = argparse.ArgumentParser(description="Choose between LDA and BERT models.")
    parser.add_argument('--model', type=str, choices=['lda', 'bert'], default='lda', help='Choose the model type.')

    args = parser.parse_args()

    data_filepath = "data/data-1716191272369.csv"
    processor = TwitterDataProcessing(filepath=data_filepath)
    print("Data preprocessing ...")
    df = processor.process()

    if args.model == 'lda':
        print("Using LDA ...")
        topics = lda_model(df=df)
    elif args.model == 'bert':
        print("Using BERTtopic ...")
        topics = bertopic_model(df=df)
    else:
        print("Invalid model choice.")
        print("Starting the topic modelling with default LDA")
        topics = lda_model(df=df)

    print('Resulting top 10 topics:')
    print(topics[:10])
