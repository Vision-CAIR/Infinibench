# Annotation Pipeline for InfiniBench Benchmark 
![annotation_pipeline](../figs/ann_pipeline_iccv.png)
## Prepare the data sources
### Data scrapping 
1) We scrapped the all the TVQA summaries from IMDB. 
2) We scrapped the all the MovieNet summaries from IMDB. 
3) We scrapped the transcripts for all the TVQA videos. 
5) We filtered out scripts for the movies that doesn't have shot subtitles from the MovieNet data.
6) We filtered out scripts for the edpisodes that doesn't exist in Long TVQA.
7) We scrapped the the spoiler questions for all the movies in movieNet.
8) We scrapped the movies durations from IMDB. 

You can see the code for scrapping the data from IMDB [here](https://github.com/Vision-CAIR/Long_video_Bench/tree/main/scrapping) but don't need to re-run it as we provide the filtered data in the benchmark sources.
### Bechmark sources : 
1) TVQA and MovieNet filtered summaries and scripts. [Download](https://huggingface.co/datasets/Vision-CAIR/InfiniBench/tree/main/sources)
2) TVQA+ annotations [Download](https://tvqa.cs.unc.edu/download_tvqa_plus.html) 
## Annotation pipeline
### Global appearance <br>
1) Download TVQA+ annotations to this directory `global_apprerance/tvqa`.
2) Filter the characters appearance in separate folders by running the following script.
```python
cd global_apprerance/tvqa
bash Run_full_pipeline.sh
```
1) Choose the best and unique outfits for each character.(humanly).
2) Run the following script to get the descriptions for the unique outfits.
```python 
python gpt4_description.py --data_path "path to the unique images folder" --output_path "path to the output folder" --api_key "GPT-4o API key"
```
1) Run the following script for question generation.
```python
python questions_generation/tvqa/global_apperance_qa_generation.py --gpt4_descriptions "path to the json file with the descriptions" --existed_episodes "existed_videos_tvqa.json"
```
### Scene transition 
```python 
python GPT-4/tvqa/python scene_transitions.py --api_key "GPT-4 API key" --scripts_folder "path to the episodes scripts folder" --output_dir "path to the output directory" --output_json "path to the output json file" --num_tasks 64
# for question generation run the following script
python questions_generation/tvqa/scene_transition_qa_generation.py --gpt4_output "path to the output json file" --existed_episodes "existed_videos_tvqa.json"
```
### Squence of character actions 
For TVQA 
```python 
python GPT-4/tvqa/character_actions.py --api_key "GPT-4 API key" --scripts_folder "path to the episodes scripts folder" --summaries_folder "path to the summaries folder" --output_dir "path to the output directory" --output_json "path to the output json file" --num_tasks 64

# for question generation run the following script
python questions_generation/tvqa/character_actions_mcq.py --gpt4_output "path to the output json file" 
```
For MovieNet 
```python 
python GPT-4/movienet/character_actions.py --api_key "GPT-4 API key" --scripts_folder "path to the movies scripts folder" --summaries_folder "path to the movies summaries folder" --output_dir "path to the output directory" --output_json "path to the output json file" --num_tasks 64
# for question generation run the following script
python questions_generation/movienet/character_actions_mcq_movienet.py --gpt4_output "path to the output json file" 
```
### Deep context understanding 
For TVQA 
```python 
python GPT-4/tvqa/context_understanding.py --api_key "GPT-4 API key" --scripts_folder "path to the episodes scripts folder" --summaries_folder "path to the summaries folder" --output_dir "path to the output directory" --output_json "path to the output json file" --num_tasks 64

# for question generation run the following script
python questions_generation/tvqa/context_understanding.py --gpt4_output "path to the output json file" 
```
For MovieNet 
```python 
python GPT-4/movienet/context_understanding.py --api_key "GPT-4 API key" --scripts_folder "path to the movies scripts folder" --summaries_folder "path to the movies summaries folder" --output_dir "path to the output directory" --output_json "path to the output json file" --num_tasks 64
# for question generation run the following script
python questions_generation/movienet/context_understanding.py --gpt4_output "path to the output json file" 
``` 
### Linking multiple events 
For TVQA 
```python 
python GPT-4/tvqa/linking_events.py --api_key "GPT-4 API key"  --summaries_folder "path to the summaries folder" --output_dir "path to the output directory" --output_json "path to the output json file" --num_tasks 64

# for question generation run the following script
python questions_generation/tvqa/linking_events.py --gpt4_output "path to the output json file" 
```
For MovieNet 
```python 
python GPT-4/movienet/linking_events.py --api_key "GPT-4 API key"  --summaries_folder "path to the movies summaries folder" --output_dir "path to the output directory" --output_json "path to the output json file" --num_tasks 64
# for question generation run the following script
python questions_generation/movienet/linking_events.py --gpt4_output "path to the output json file" 
```
### Temporal events 
For TVQA 
```python 
python GPT-4/tvqa/temporal_events.py --api_key "GPT-4 API key" --scripts_folder "path to the episodes scripts folder" --output_dir "path to the output directory" --output_json "path to the output json file" --num_tasks 64

# for question generation run the following script
python questions_generation/tvqa/temporal_events_qa_generation.py --gpt4_output "path to the output json file" 
```
For MovieNet 
```python 
python GPT-4/movienet/temporal_events.py --api_key "GPT-4 API key" --scripts_folder "path to the movies scripts folder" --output_dir "path to the output directory" --output_json "path to the output json file" --num_tasks 64
# for question generation run the following script
python questions_generation/movienet/temporal_events_qa_generation.py --gpt4_output "path to the output json file" 
```
### Movies spoiler questions 
```python 
python questions_generation/spoiler_questions.py --scrapped_spoiler_questions "path to the scrapped spoiler questions"
```
### Summarization 
```python
python questions_generation/summarization_skill.py --summarization_movienet_json "path to json file of movienet summaries" --summarization_tvqa_json "path to json file of tvqa summaries" --api_key "GPT-4 API key"
```

### Local visual and context understanding 
We converted the questions of the validation split from the original TVQA to Long form questions here 
`process_tvqa_videos/tvqa_val_edited.json`
```python 
python questions_generation/long_tvqa_questions.py --tvqa_val_edited "process_tvqa_videos/tvqa_val_edited.json"
```

# Acknowledgements
[InsightFace](https://github.com/deepinsight/insightface/tree/master/python-package)<br>
[Yolo-11](https://github.com/ultralytics/ultralytics)<br>