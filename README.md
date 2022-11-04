# SDU@AAAI-23 - Shared Task 1: Jargon/Terminology Detection

This repository contains the terminology detection training and development sets along with the evaluation scripts for the [Jargon/Terminology Detection task at SDU@AAAI-23](https://sites.google.com/view/sdu-aaai23/shared-task)

# Dataset

The dataset folder contains two files `train.json` and `dev.json` containing the training and development sets, respectively.

Both files have the same structure:

```
{
  "cs": {
      doc_id: [
          {
              "tokens": [
                "token_1",
                "token_2",
                ...,
                "token_n"
              ],
              "term_spans" [
                [start_1, end_1],
                ...,
                [start_m, end_m]
              ]
           },
           {
            ...
           },
         ...
      ],
  }
  "phys": {
    ...
  },
  "econ": {
    ...
  }
}
```
That is, the highest-level `dictionary` keys indentify each domain: `Computer Science - cs`, `Economics - econ`, and `Physics - phys`. Then, within each domain, documents are identified by their `doc_id` and contain a `list` of sentences. Each sentence is represented by a `dictionary` with two items: `tokens` contains a `list` of tokenized words and symbols and `term_spans` contains a `list` of boundary pairs (`start`, `end`) for each labeled term.

# Evaluation

To evaluate the predictions, run the following command:

`python scorer.py -g path/to/gold.json -p path/to/predictions.json`

The `path/to/gold.json` and `path/to/predictions.json` should be replaced with the real paths to the gold file (e.g., `dataset/dev.json` for evaluation on development set) and predictions file (i.e., the predictions generated by your system. Note that it should be in the same format as `dataset/dev.json` or `dataset/train.json` files). The official evaluation metrics are precision, recall and F1 for terminology predictions. 

# Participation
In order to participate, please first fill out [this form](https://forms.gle/ks3snBYXgoTQr3iH7) to register for the shared task. The team name that is provided in this form will be used in subsequent submissions and communications. 

The shared task is organized as a CodaLab/Kaggle [competition]().

There are two separate phases:
* _Development Phase_: In this phase, the participants will use the training/development sets provided in this repository to design and develop their models.
* _Evaluation Phase_: Ten days before the system runs are due (January 4th, 2023), the test set for the task is released in the GitHub repository. The test set has the same distribution and format as the development set. Run your model on the provided test set and save the prediction results in a JSON file with the same format as the "predictions.json" file. Name the prediction file "output.json" and submit it to the CodaLab/Kaggle [competition page]().

For more information, see [SDU@AAAI-23]().

# Licences
The dataset provided for this shared task is licensed under [CC BY-NC-SA 4.0 international license](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode), and the evaluation scrip is licensed under MIT License.
