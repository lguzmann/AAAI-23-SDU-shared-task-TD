import json
import argparse

from tqdm import tqdm
from seqeval.metrics import f1_score, precision_score, recall_score


def run_evaluation(args):
    verbose = args.v

    gold = {"cs": [], "econ": [], "phys": []}
    prediction = {"cs": [], "econ": [], "phys": []}
   
    if verbose:
        print("Loading gold labels...")
    with open(args.g) as file:
        full_dict = json.load(file)
        for domain in full_dict:
            domain_dict = full_dict[domain]
            for doc in domain_dict:
                doc_lst = domain_dict[doc]
                for sentence in doc_lst:
                    tokens = sentence['tokens']
                    spans = sentence['term_spans']

                    #Creating a list of labels with the length of the token sequence
                    labels = ['O'] * len(tokens)
                    
                    #Using the spans to label the tokens
                    if len(spans) > 0:
                        for span in spans:
                            labels[span[0]: span[1]] = 'B'
                    
                    #Appending to the gold dict
                    gold[domain].append(labels)

    if verbose:
        print("Lading predicted labels...")
    with open(args.p) as file:
        full_dict = json.load(file)
        for domain in full_dict:
            domain_dict = full_dict[domain]
            for doc in domain_dict:
                doc_lst = domain_dict[doc]
                for sentence in doc_lst:
                    tokens = sentence['tokens']
                    spans = sentence['term_spans']

                    #Creating a list of labels with the length of the token sequence
                    labels = ['O'] * len(tokens)

                    #Using the spans to label the tokens
                    if len(spans) > 0:
                        for span in spans:
                            labels[span[0]: span[1]] = 'B'

                    #Appending to the prediction dict
                    prediction[domain].append(labels)

    print(f"\n{'Domain':<10}{'Precision':<10}{'Recall':<10}F1 Score")
    
    full_gold = []
    full_pred = []

    for domain in gold:
        full_gold += gold[domain]
        full_pred += prediction[domain]

        if verbose:
            f1 = f1_score(gold[domain], prediction[domain], average = 'micro')
            precision = precision_score(gold[domain], prediction[domain], average = 'micro')
            recall = recall_score(gold[domain], prediction[domain], average = 'micro')
    
            print(f"{domain:<10}{precision:<10}{recall:<10}{f1}")

    f1 = f1_score(full_gold, full_pred, average = 'micro')
    precision = precision_score(full_gold, full_pred, average = 'micro')
    recall = recall_score(full_gold, full_pred, average = 'micro')

    print(f"{'Overall':<10}{precision:<10}{recall:<10}{f1}\n")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', type=str,
                        help='Gold file path')
    parser.add_argument('-p', type=str,
                        help='Prediction file path')
    parser.add_argument('-v', dest='v',
                        default=False, action='store_true',
                        help="Verbose Evaluation")


    args = parser.parse_args()
    run_evaluation(args)
    #print('Official Scores:')
    #print('P: {:.2%}, R: {:.2%}, F1: {:.2%}'.format(p,r,f1))
