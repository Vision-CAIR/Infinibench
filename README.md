# <img src="figs/icon_png.png" width=30> InfiniBench: A Benchmark for Large Multi-Modal Models in Long-Form Movies and TV Shows

<font size=3><div align='center' > [[<img src="figs/icon_png.png" width=18> Project Page](https://vision-cair.github.io/InfiniBench/)] [[📝 arXiv Paper](https://arxiv.org/abs/2406.19875)] [[🤗 Download](https://huggingface.co/datasets/Vision-CAIR/InfiniBench/tree/main)]</div></font>

![InfiniBench teaser figure](figs/teaser_fig.png)
<strong>InfiniBench skill set comprising eight skills. The right side represents skill categories and question types, while the left side provides examples of both multiple-choice (MCQ) and open-ended questions.</strong>

# Overview:
Understanding long-form videos, such as movies and TV episodes ranging from tens of minutes to two hours, remains a major challenge for multi-modal models. Existing benchmarks often fall short in testing the full range of cognitive skills needed to process these temporally rich and narratively complex inputs. We introduce InfiniBench, a comprehensive benchmark designed to rigorously evaluate the capabilities of models in long video understanding.
InfiniBench offers:
**(1) Over 1,000 hours of video content, with an average video length of 52.59 minutes,(2) The largest set of question-answer pairs for long video comprehension, totaling around \totalSampleNumber, (3) Eight diverse skills that span both grounding-based (e.g., scene transitions, character actions) and reasoning-based (e.g., deep context, multi-event linking) understanding, and (4) Rich annotation formats, including both multiple-choice and open-ended questions.**
We conduct an in-depth evaluation across both commercial (GPT-4o, Gemini 1.5 Flash) and open-source (Qwen2.5-VL, InternVL2.5) vision-language models. 
Results reveal that current models remain far from solving long video understanding: on grounding-based skills, the top open-source model (Qwen2.5-VL) and GPT-4o achieve only 39.4\% and 48.1\% accuracy, respectively. 
Interestingly, several models achieve non-trivial performance using only the movie or episode title, without watching the video, revealing a reliance on pre-trained world knowledge that partially compensates for the absence of visual or temporal understanding.
These findings highlight critical gaps in current approaches and underscore the need for models that truly engage with long visual narratives.

# Leaderboard for top commercial and open souce models:
<table>
  <thead>
    <tr>
      <th rowspan="2">Models</th>
      <th rowspan="2">Frame Rate</th>
      <th colspan="4" style="text-align:center; border-bottom: 1px solid #E4EAFF;">Grounding Skills</th>
      <th colspan="4" style="text-align:center; border-bottom: 1px solid #FFF2CC;">Reasoning Skills</th>
      <th rowspan="2">Avg. Acc.</th>
      <th rowspan="2">Avg. Score</th>
    </tr>
    <tr>
      <th>Global Appearance</th>
      <th>Scene Transitions</th>
      <th>Character Actions</th>
      <th>Chronological Understanding</th>
      <th>Summarization</th>
      <th>Deep Context Understanding</th>
      <th>Spoiler Understanding</th>
      <th>Linking Events</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color:#92a2fc;">
      <td>Baseline Random</td>
      <td>--</td>
      <td>16.68</td>
      <td>16.66</td>
      <td>16.14</td>
      <td>41.51</td>
      <td>--</td>
      <td>--</td>
      <td>--</td>
      <td>--</td>
      <td>22.20</td>
      <td>--</td>
    </tr>
    <tr>
      <td>GPT-4o</td>
      <td>250 FPV</td>
      <td>44.51</td>
      <td>47.93</td>
      <td>36.07</td>
      <td>68.85</td>
      <td>3.49</td>
      <td>3.39</td>
      <td>2.67</td>
      <td>3.45</td>
      <td>49.34</td>
      <td>3.25</td>
    </tr>
    <tr>
      <td>Gemini-1.5-flash</td>
      <td>-</td>
      <td>42.10</td>
      <td>31.63</td>
      <td>37.82</td>
      <td>56.41</td>
      <td>3.24</td>
      <td>2.55</td>
      <td>2.05</td>
      <td>3.33</td>
      <td>41.99</td>
      <td>2.79</td>
    </tr>
    <tr>
      <td>Qwen2.5VL</td>
      <td>250 FPV</td>
      <td>34.99</td>
      <td>36.45</td>
      <td>35.09</td>
      <td>51.57</td>
      <td>1.26</td>
      <td>2.35</td>
      <td>1.73</td>
      <td>3.15</td>
      <td>39.53</td>
      <td>2.12</td>
    </tr>
    <tr>
      <td>Qwen2VL</td>
      <td>250 FPV</td>
      <td>29.99</td>
      <td>37.54</td>
      <td>36.86</td>
      <td>50.85</td>
      <td>0.67</td>
      <td>2.07</td>
      <td>1.41</td>
      <td>2.76</td>
      <td>38.81</td>
      <td>1.73</td>
    </tr>
    <tr>
      <td>LongVU</td>
      <td>250 FPV</td>
      <td>38.46</td>
      <td>22.69</td>
      <td>28.97</td>
      <td>45.13</td>
      <td>0.20</td>
      <td>1.10</td>
      <td>0.71</td>
      <td>1.37</td>
      <td>33.81</td>
      <td>0.84</td>
    </tr>
    <tr>
      <td>LLaVA-OneVision</td>
      <td>128 FPV</td>
      <td>33.00</td>
      <td>25.02</td>
      <td>24.83</td>
      <td>45.91</td>
      <td>0.49</td>
      <td>1.78</td>
      <td>1.30</td>
      <td>2.51</td>
      <td>32.19</td>
      <td>1.52</td>
    </tr>
    <tr>
      <td>InternLM-XComposer-2.5-OL</td>
      <td>16 FPW</td>
      <td>27.17</td>
      <td>24.37</td>
      <td>30.09</td>
      <td>46.68</td>
      <td>0.37</td>
      <td>1.21</td>
      <td>0.61</td>
      <td>2.03</td>
      <td>32.08</td>
      <td>1.06</td>
    </tr>
    <tr>
      <td>InternVL2.5</td>
      <td>128 FPV</td>
      <td>29.84</td>
      <td>25.35</td>
      <td>26.41</td>
      <td>45.58</td>
      <td>0.65</td>
      <td>1.48</td>
      <td>1.06</td>
      <td>2.22</td>
      <td>31.80</td>
      <td>1.35</td>
    </tr>
     <tr >
      <td>InternVL2</td>
      <td>128 FPV</td>
      <td>24.60</td>
      <td>21.98</td>
      <td>25.00</td>
      <td>44.63</td>
      <td>0.69</td>
      <td>1.68</td>
      <td>1.25</td>
      <td>2.47</td>
      <td>29.05</td>
      <td>1.52</td>
    </tr>
    <tr >
      <td>LLaMA-VID</td>
      <td>1 FPS</td>
      <td>17.37</td>
      <td>17.06</td>
      <td>18.25</td>
      <td>41.74</td>
      <td>1.58</td>
      <td>2.00</td>
      <td>1.49</td>
      <td>2.40</td>
      <td>23.61</td>
      <td>1.87</td>
    </tr>
    <tr>
      <td>Goldfish</td>
      <td>45 FPW</td>
      <td>10.30</td>
      <td>2.82</td>
      <td>20.87</td>
      <td>40.14</td>
      <td>0.77</td>
      <td>2.36</td>
      <td>1.85</td>
      <td>3.01</td>
      <td>18.53</td>
      <td>2.00</td>
    </tr>
    <tr>
      <td>MiniGPT4-video</td>
      <td>45 FPV</td>
      <td>2.33</td>
      <td>1.09</td>
      <td>2.36</td>
      <td>39.86</td>
      <td>0.05</td>
      <td>0.54</td>
      <td>0.75</td>
      <td>0.89</td>
      <td>11.41</td>
      <td>0.56</td>
    </tr>
  </tbody>
</table>
<p><strong>InfiniBench leaderboard</strong> across eight skills. FPV (Frames Per Video), FPS (Frames Per Second), and FPW (Frames Per Window) are reported. All models in this evaluation utilize <strong>subtitles</strong>.</p>

# Benchmark statistics:
![benchmark_statistics_1](figs/full_data_statistics.png)
<strong>InfiniBench skills statistics. (A) Number of questions per skill, (B) Number of videos per skill, and (C) Average video duration per skill</strong>
# Videos source statistics:
![benchmark_statistics_2](figs/shows_vs_movies_statistics.png)
<strong>Comparison between TV shows and Movies. (A) shows the number of questions, (B) represents the number of videos, (C) represents the Total video durations, and (D) shows The Minimum, Maximum, and average video duration for each video source</strong>


# Download The Benchmark
We are only provide annotations for already extisting videos datasets, namely TVQA and MovieNet.
To make it easier for you to use the benchmark, we have preprocessed the videos and subtitles for both TVQA and MovieNet datasets.<br>
You can directly download the preprocessed version from the table below. <br>
| Split | TVshows Videos | TVshows subtitles| MovieNet Videos | MovieNet subtitles | Annotations |
|-------|----------------|------------------|------------------|--------------------|-------------|
| Test  | [Download](https://huggingface.co/datasets/Vision-CAIR/InfiniBench/resolve/main/tvqa_mp4_videos_test.zip) | [Download](https://huggingface.co/datasets/Vision-CAIR/InfiniBench/resolve/main/tvqa_subtitles_test.zip) | [Download](https://huggingface.co/datasets/Vision-CAIR/InfiniBench/resolve/main/movienet_mp4_videos_test.zip) | [Download](https://huggingface.co/datasets/Vision-CAIR/InfiniBench/resolve/main/movienet_subtitles_test.zip) | [Download](https://huggingface.co/datasets/Vision-CAIR/InfiniBench/tree/main/Benchmark_annotations) |
| Train | [Download](https://huggingface.co/datasets/Vision-CAIR/InfiniBench/resolve/main/tvqa_mp4_videos_train.zip) | [Download](https://huggingface.co/datasets/Vision-CAIR/InfiniBench/resolve/main/tvqa_subtitles_train.zip) | [Download](https://huggingface.co/datasets/Vision-CAIR/InfiniBench/resolve/main/movienet_mp4_videos_train.zip) | [Download](https://huggingface.co/datasets/Vision-CAIR/InfiniBench/resolve/main/movienet_subtitles_train.zip) | [Download](https://huggingface.co/datasets/Vision-CAIR/InfiniBench/tree/main/Benchmark_annotations) |

OR <br>

You can download the original data and preprocess it using the scripts provided in this repository.
## TVQA videos <br>
Download the original TVQA videos for short videos from [here](https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/download_tvqa.html)<br>
Also download the original TVQA annotations from [here](https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/download_tvqa.html) <br>

### Adding audio to the short clips
Add the audio to the short clips 
```python
```
### Convert the short clips to long-form videos
Run the following commmand to convert the videos to long-form videos.<br>
```python
python data_genration/videos_preprocessing/convert_tvqa_from_short_to_long.py --train_path "path to the training annotation" --val_path "path to the validation annotation" --root_dir "path to the short clips directory" --full_videos_dir "path to save the full video episodes"
```

this script will output the full video episodes frames in the full_videos_dir, then you can use the following script to convert the frames to .mp4 files. <br>

Run the following script
```python
python data_genration/videos_preprocessing/convert_to_mp4_format.py --video_frames_dir "path to the long videos frames" --output_dir "path to save the MP4 videos" --source "tvqa" --fps 3 
```
### Concate the subtitles to the long-form videos
```python 
```

## MovieNet Data <br>
Dowlnoad the original MovieNet data from [here](https://opendatalab.com/OpenDataLab/MovieNet/tree/main/raw) <br>
Filter out the movies that doesn't have shot subtitles<br>
Run the following script to filter movienet<br>
```python
python data_genration/filter_movienet.py
```
To get the video .mp4 files,Run the following script to the raw data 
```python
# to generare movies with the original frame rate use original_fps = True
python data_genration/videos_preprocessing/convert_to_mp4_format.py --video_frames_dir "path to the long videos frames" --output_dir "path to save the MP4 videos" --source "movienet" --original_fps --movies_has_subtitles "movies_has_subtitles.json" --movies_durations "movies_durations.json" 
# to generate movies with 1 fps use original_fps = False and fps = 1 but take care that the video duration will be different from the original duration 
python data_genration/videos_preprocessing/convert_to_mp4_format.py --video_frames_dir "path to the long videos frames" --output_dir "path to save the MP4 videos" --source "movienet" --fps 1 --movies_has_subtitles "movies_has_subtitles.json" --movies_durations "movies_durations.json" 
```
Subtitles are already provided in the original MovieNet data, so you don't need to do anything for subtitles. <br>
## Benchmark annotations pipeline
View the [data_genration/README.md](data_genration/README.md) for the full annotation pipeline details <br>

# Citation
If you're using InfiniBench in your research or applications, please cite using this BibTeX:
```
@misc{ataallah2024infinibenchcomprehensivebenchmarklarge,
      title={InfiniBench: A Comprehensive Benchmark for Large Multimodal Models in Very Long Video Understanding}, 
      author={Kirolos Ataallah and Chenhui Gou and Eslam Abdelrahman and Khushbu Pahwa and Jian Ding and Mohamed Elhoseiny},
      year={2024},
      eprint={2406.19875},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.19875}, 
}
```

# License
This repository is under [BSD 3-Clause License](LICENSE.md).