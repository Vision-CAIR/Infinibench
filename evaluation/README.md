# Evaluation
To use our evaluation scrip for accuracy and GPT4 score you should prepare each skill prediction file in the following format.
```python 
# for multiple choice questions
[
    {"Q":"question",  "A","answer", "pred":"model_pred","options_str":"option 0 : option sentence \n option 1 option sentence \n ...","answer_idx":"correct option index"}  ,
    {"Q":"question",  "A","answer", "pred":"model_pred","options_str":"option 0 : option sentence \n option 1 option sentence \n ...","answer_idx":"correct option index"}  ,
    {"Q":"question",  "A","answer", "pred":"model_pred","options_str":"option 0 : option sentence \n option 1 option sentence \n ...","answer_idx":"correct option index"}  ,
    ... 
]

# for open ended questions 
[
    {"Q":"question",  "A","answer", "pred":"model_pred"}  ,
    {"Q":"question",  "A","answer", "pred":"model_pred"}  ,
    {"Q":"question",  "A","answer", "pred":"model_pred"}  ,
    ... 
]

```
Then run the following script for accuracy evaluation for the skills that has multiple choice questions 
```bash
# set the parameters in the script
bash evaluation/GPT4_eval/gpt4_accuracy.sh 
```
For the skills that has open-ended questions run the following script to get the GPT4 score
```bash
# set the parameters in the script
bash evaluation/GPT4_eval/gpt4_score.sh 
```
