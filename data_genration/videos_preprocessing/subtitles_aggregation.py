import os 
import json 
import pysrt 
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--tvqa_subtitles_folder",type=str,help="TVQA clips subtitles folder")
parser.add_argument("--tvqa_videos_path",type=str,help="The path to the tvqa videos")
parser.add_argument("--output_dir",type=str,help="The path to save the aggregated subtitles")
args=parser.parse_args()
all_tvqa_subtitles_folder=args.tvqa_subtitles_folder
all_tvqa_videos_path=args.tvqa_videos_path
save_aggregated_subtitles_path=args.output_dir
tvqa_data_path="tvqa_full_data.json"
mapping={"Grey's Anatomy":"grey", 'How I Met You Mother':"met", 'Friends':"friends", 'The Big Bang Theory':"bbt", 'House M.D.':"house", 'Castle':"castle"} 

with open(tvqa_data_path, 'r') as file:
    tvqa_data = json.load(file)

def time_to_milliseconds(time_str):
    # Convert time format "hh:mm:ss.sss" to milliseconds
    h, m, s = map(float, time_str.split(':'))
    return int((h * 3600 + m * 60 + s) * 1000)

def edit_subtitle_file (subtitle_path, added_time_ms,index):
    # add the added_time that in millseconds to each subtitle in the subtitle file
    subs = pysrt.open(subtitle_path)
    # convert the added_time_ms to datetime object 
    # added_time=datetime.timedelta(milliseconds=added_time_ms)
    for sub in subs:
        sub.index+=index
        new_start_ms = time_to_milliseconds(str(sub.start.to_time()))+added_time_ms
        new_end_ms = time_to_milliseconds(str(sub.end.to_time()))+added_time_ms
        sub.start = pysrt.SubRipTime(milliseconds=new_start_ms)
        sub.end = pysrt.SubRipTime(milliseconds=new_end_ms)
    return subs

subtitles_errors={}
for show in tqdm(tvqa_data,desc="shows"):
    mapped_show=mapping[show]
    # if mapped_show != "friends":
    #     continue
    for season in tqdm(tvqa_data[show],desc="seasons"):
        # if season != "season_1":
        #     continue
        for episode in tqdm(tvqa_data[show][season],desc="episodes"):
            # if episode != "episode_1":
            #     continue
            clips=tvqa_data[show][season][episode]['clips']
            # print("Clips ",clips)
            # each clip has its own subtitles , we need to collect all the subtitles of the episode 
            prev_clip_duration=0 
            episode_subtitles=pysrt.SubRipFile()
            prev_index=0
            for clip in clips:
                clip_path=os.path.join(all_tvqa_videos_path,mapped_show+"_frames",clip+".mp4")
                clip_subtitle_path=os.path.join(all_tvqa_subtitles_folder,clip+".srt")
                try:
                    subs=edit_subtitle_file(clip_subtitle_path, prev_clip_duration,prev_index)
                    prev_index+=len(subs)
                    episode_subtitles.extend(subs)
                except Exception as e:
                    # these clips have no subtitles but we can ignore them and continue the aggregation
                    print("Error :", e)
                    subtitles_errors[clip_subtitle_path]=str(e)             
                clip_duration=VideoFileClip(clip_path).duration
                clip_duration_ms=clip_duration*1000
                prev_clip_duration+=clip_duration_ms
                
            
            # save the episode subtitles
            episode_subtitles_path=os.path.join(save_aggregated_subtitles_path,mapped_show,season)
            os.makedirs(episode_subtitles_path,exist_ok=True)
            episode_subtitles_path=os.path.join(episode_subtitles_path,episode+".srt")
            # save the subtitles , convert the episode_subtitles to <class 'pysrt.srtfile.SubRipFile'> object
            episode_subtitles.save(episode_subtitles_path)
            # print("saved ",episode_subtitles_path)
            # episode_subtitles.save("kero.srt")

# save the errors
with open("subtitles_errors.json", 'w') as file:
    json.dump(subtitles_errors, file, indent=4)
                
                
                