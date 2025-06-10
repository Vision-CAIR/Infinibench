# InfiniBench Benchmark Sources
## videos 
### TVQA videos <br>
Download the original TVQA videos for short videos from [here](https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/download_tvqa.html)<br>
Also download the original TVQA subtitles from [here](https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/download_tvqa.html) <br>

### Convert the short clips to long-form videos
Run the following commmand to convert the videos to long-form videos.<br>
```bash
python data_genration/videos_preprocessing/convert_tvqa_from_short_to_long.py --train_path "path to the training annotation" --val_path "path to the validation annotation" --root_dir "path to the short clips directory" --full_videos_dir "path to save the full video episodes"
```

This script will output the full video episodes frames in the full_videos_dir, then you can use the following script to convert the frames to .mp4 files. <br>

Run the following script
```bash
python data_genration/videos_preprocessing/convert_to_mp4_format.py --video_frames_dir "path to the long videos frames" --output_dir "path to save the MP4 videos" --source "tvqa" --fps 3 
```
### Concate the subtitles to the long-form videos
```bash 
python data_genration/videos_preprocessing/subtitles_aggregation.py --tvqa_subtitles_folder "path to the TVQA subtitles folder" --output_dir "path to save the aggregated subtitles" tvqa_videos_path "Path to the long-form videos directory"

```

## MovieNet Data <br>
Dowlnoad the original MovieNet data from [here](https://opendatalab.com/OpenDataLab/MovieNet/tree/main/raw) <br>
Filter out the movies that doesn't have shot subtitles<br>
Run the following script to filter movienet<br>
```bash
python data_genration/filter_movienet.py
```
To get the video .mp4 files,Run the following script to the raw data 
```bash
# to generare movies with the original frame rate use original_fps = True
python data_genration/videos_preprocessing/convert_to_mp4_format.py --video_frames_dir "path to the long videos frames" --output_dir "path to save the MP4 videos" --source "movienet" --original_fps --movies_has_subtitles "movies_has_subtitles.json" --movies_durations "movies_durations.json" 
# to generate movies with 1 fps use original_fps = False and fps = 1 but take care that the video duration will be different from the original duration 
python data_genration/videos_preprocessing/convert_to_mp4_format.py --video_frames_dir "path to the long videos frames" --output_dir "path to save the MP4 videos" --source "movienet" --fps 1 --movies_has_subtitles "movies_has_subtitles.json" --movies_durations "movies_durations.json" 
```
Subtitles are already provided in the original MovieNet data, so you don't need to do anything for subtitles. <br>

## Transcripts and Summaries
### Data scrapping 
We need the summaries and transcripts for the TV shows and movies in the benchmark. <br>
MovieNet provides the transcripts for the movies, but we need to scrape the transcripts for the six TV shows.
We scrapped the transcripts from  [foreverdreaming](https://transcripts.foreverdreaming.org/) which provides transcripts for many TV shows and already have the six TV shows we need. <br>
As **summarization** is a key skill in the benchmark, we also scraped the summaries for the movies and TV shows from [IMDb](https://www.imdb.com/). <br>
We used the following steps to prepare the data:
1. Filter out movies from MovieNet that did not have corresponding shot-level subtitles.
2. Collect all available summaries for movies and TV shows from IMDb.
3. Scraped transcripts for the TV shows from [foreverdreaming](https://transcripts.foreverdreaming.org/).
4. Scraped spoiler questions for the filtered movies from MovieNet.
5. Scraped movie durations from IMDb to be able to construct the long-form videos.

You can see the scrapping scripts for the data from IMDB and foreverdreaming [here](https://github.com/Vision-CAIR/Infinibench/tree/main/data_genration/scrapping) but don't need to re-run it as we provide the filtered data in the benchmark sources [Download](https://huggingface.co/datasets/Vision-CAIR/InfiniBench/tree/main/sources_filtered)

# Annotation Pipeline 
![annotation_pipeline](../figs/ann_pipeline_iccv.png)
### Global Appearance <br>
Steps:
1. Scrap the unique images of the characters of the TV shows from the IMDB.
2. Filter out the characters thet appear in less than 20 episodes.
3. Apply Yolo-11 to detect the characters in the images and segment them , use face recognition recognize the character name and save each character in a separate folder.
```bash
python data_genration/global_apprerance/tvqa/character_detection.py --videos_path "path to the TV shows videos" --output_dir "path to save the character images" --show "show folder name" --season "season number"
```
4. In each folder, Filter out similar images using dinov2.0 embeddings with threshold 0.6.
```bash
python data_genration/global_apprerance/tvqa/filter_similar_images.py --characerts_path "path to the saved characters folder" --output_path "path to save the unique images" 
```
5. Generate descriptions for the unique outfits using GPT-4o.

```bash 
export OPENAI_API_KEY="Your GPT-4o API key"

python data_genration/global_apprerance/tvqa/generate_descriptions_and_transitions.py --filtered_images_folder "path to the unique images folder out from dino filtering" --output_path "path to the output folder" --original_images_dir "path to the original images folder before dino filtering" 
```
6. Question generation for the global appearance skill using GPT-4o.
```bash
python data_genration/global_apprerance/tvqa/global_apperance_qa_generation.py --annotations_folder "path to the output folder of GPT4o annotations from the previous step" --output_json "save path for the generated questions json file" 
```
### Scene Transition 
```bash 
python data_genration/GPT4o-skills_generation/tvqa/scene_transitions.py --api_key "GPT-4 API key" --scripts_folder "path to the episodes scripts folder" --output_dir "path to the output directory" --output_json "path to the output json file" --num_tasks 64
# for question generation run the following script
python data_genration/questions_generation/tvqa/scene_transition_qa_generation.py --gpt4_output "path to the output json file" --existed_episodes "existed_videos_tvqa.json"
```
### Character Actions 
For TVQA 
```bash 
python data_genration/GPT4o-skills_generation/tvqa/character_actions.py --api_key "GPT-4 API key" --scripts_folder "path to the episodes scripts folder" --summaries_folder "path to the summaries folder" --output_dir "path to the output directory" --output_json "path to the output json file" --num_tasks 64

# for question generation run the following script
python data_genration/questions_generation/tvqa/character_actions_mcq.py --gpt4_output "path to the output json file" 
```
For MovieNet 
```bash 
python data_genration/GPT4o-skills_generation/movienet/character_actions.py --api_key "GPT-4 API key" --scripts_folder "path to the movies scripts folder" --summaries_folder "path to the movies summaries folder" --output_dir "path to the output directory" --output_json "path to the output json file" --num_tasks 64
# for question generation run the following script
python data_genration/questions_generation/movienet/character_actions_mcq_movienet.py --gpt4_output "path to the output json file" 
```
### Deep context understanding 
For TVQA 
```bash 
python data_genration/GPT4o-skills_generation/tvqa/context_understanding.py --api_key "GPT-4 API key" --scripts_folder "path to the episodes scripts folder" --summaries_folder "path to the summaries folder" --output_dir "path to the output directory" --output_json "path to the output json file" --num_tasks 64

# for question generation run the following script
python data_genration/questions_generation/tvqa/context_understanding.py --gpt4_output "path to the output json file" 
```
For MovieNet 
```bash 
python data_genration/GPT4o-skills_generation/movienet/context_understanding.py --api_key "GPT-4 API key" --scripts_folder "path to the movies scripts folder" --summaries_folder "path to the movies summaries folder" --output_dir "path to the output directory" --output_json "path to the output json file" --num_tasks 64
# for question generation run the following script
python data_genration/questions_generation/movienet/context_understanding.py --gpt4_output "path to the output json file" 
``` 
### Linking multiple events 
For TVQA 
```bash 
python data_genration/GPT4o-skills_generation/tvqa/linking_events.py --api_key "GPT-4 API key"  --summaries_folder "path to the summaries folder" --output_dir "path to the output directory" --output_json "path to the output json file" --num_tasks 64

# for question generation run the following script
python data_genration/questions_generation/tvqa/linking_events.py --gpt4_output "path to the output json file" 
```
For MovieNet 
```bash 
python data_genration/GPT4o-skills_generation/movienet/linking_events.py --api_key "GPT-4 API key"  --summaries_folder "path to the movies summaries folder" --output_dir "path to the output directory" --output_json "path to the output json file" --num_tasks 64
# for question generation run the following script
python data_genration/questions_generation/movienet/linking_events.py --gpt4_output "path to the output json file" 
```
### Temporal events 
For TVQA 
```bash 
python data_genration/GPT4o-skills_generation/tvqa/temporal_events.py --api_key "GPT-4 API key" --scripts_folder "path to the episodes scripts folder" --output_dir "path to the output directory" --output_json "path to the output json file" --num_tasks 64

# for question generation run the following script
python data_genration/questions_generation/tvqa/temporal_events_qa_generation.py --gpt4_output "path to the output json file" 
```
For MovieNet 
```bash 
python data_genration/GPT4o-skills_generation/movienet/temporal_events.py --api_key "GPT-4 API key" --scripts_folder "path to the movies scripts folder" --output_dir "path to the output directory" --output_json "path to the output json file" --num_tasks 64
# for question generation run the following script
python data_genration/questions_generation/movienet/temporal_events_qa_generation.py --gpt4_output "path to the output json file" 
```
### Movies spoiler questions 
```bash 
python data_genration/questions_generation/spoiler_questions.py --scrapped_spoiler_questions "path to the scrapped spoiler questions"
```
### Summarization 
```bash
python data_genration/questions_generation/summarization_skill.py --summarization_movienet_json "path to json file of movienet summaries" --summarization_tvqa_json "path to json file of tvqa summaries" --api_key "GPT-4 API key"
```

# Acknowledgements
[InsightFace](https://github.com/deepinsight/insightface/tree/master/python-package)<br>
[Yolo-11](https://github.com/ultralytics/ultralytics)<br>