import ffmpeg
import moviepy.editor as mp
import librosa
import os
import numpy as np
import random
import soundfile as sf

clip_dir = 'clips'

crop_factor = [
    [],
    [0, 0, 1, 1],
    [0, 0.25, 1, 0.75],
    [0, 1/3, 1, 2/3],
    [0.25, 0.75, 0.25, 0.75]
]
reshape_format = [
    [],
    [1, 1],
    [2, 1],
    [3, 1],
    [2, 2]
]

batch_size = 16

def main(fn_bgm):
    # load audio
    # .wav format
    x_bgm, sr_bgm = librosa.load(fn_bgm)

    # beat track
    tempo, beat_times_bgm = librosa.beat.beat_track(x_bgm, sr=sr_bgm, start_bpm=200, units='time')
    print('beats extracted:', len(beat_times_bgm))

    # add heading 0
    beat_times_bgm = np.insert(beat_times_bgm, 0, 0)
    beat_times_bgm = np.append(beat_times_bgm, librosa.get_duration(x_bgm, sr_bgm))

    # collect clips
    clip_names = os.listdir(clip_dir)
    clip_names = list(filter(lambda fn: fn.endswith('.mp4'), clip_names))

    clips = dict()

    # fill in all beats
    for name in clip_names:
        print('\rsegmenting', name, end='')
        clips[name] = []

        # load clip
        # .mp4 format
        temp_clip = mp.VideoFileClip(os.path.join(clip_dir, name))

        # extract split point
        l1, sr = librosa.load(os.path.join(clip_dir, name[:-4] + '.wav'))
        _, beat_times = librosa.beat.beat_track(l1, sr=sr, start_bpm=50, units='time')
        beat_times = np.insert(beat_times, 0, 0)

        temp_clips = []
        # segment the clip
        for j in range(min(len(beat_times) - 1, 64)):
            temp_clips.append(temp_clip.subclip(beat_times[j], beat_times[j + 1]))

            if len(temp_clips) == batch_size:
                clips[name].append(temp_clips)
                temp_clips = []

        # if len(temp_clips):
        #     clips[name].append(temp_clips)

    print('\rsegmentation done')

    # fill in bgm beats
    flat_clips = []
    i = 0
    n = 0
    while i < len(beat_times_bgm):
        print('\rconstructing beat at', i, end='')
        # create collage of clips
        keys = list(filter(lambda k: len(clips[k]), clips.keys()))

        # randomly select clips to collage
        l = min(len(crop_factor) - 1, len(keys))
        if not l:
            break
        # n = random.choices(range(1, l + 1, 1), weights=[4, 2, 1, 1][:l])[0]
        n = n % (len(crop_factor) - 1) + 1
        if n > l:
            n = l
        keys = random.sample(keys, k=n)

        collage_clips = []
        for key in keys:
            # for each batch
            temp_clips = clips[key].pop()
            if len(beat_times_bgm) - 1 - i < batch_size:
                temp_clips = temp_clips[:len(beat_times_bgm) - 1 - i]
            for j, temp_clip in enumerate(temp_clips):
                # set start end time
                s = beat_times_bgm[i + j + 1] - beat_times_bgm[i + j]
                t = temp_clip.duration

                if t < s:
                    # extend the segment
                    temp_clips[j] = temp_clip.fx(mp.vfx.speedx, final_duration=s)
                else:
                    # trim the segment
                    temp_clips[j] = temp_clip.subclip(t - s, t)

            collage_clips.append(mp.concatenate_videoclips(temp_clips))

        # crop clip to fit
        fac = crop_factor[n]
        l1, l2 = collage_clips[0].size
        for j, temp_clip in enumerate(collage_clips):
            collage_clips[j] = temp_clip.crop(x1=l1 * fac[0], y1=l2 * fac[1], x2=l1 * fac[2], y2=l2 * fac[3])

        collage_clips = np.array(collage_clips).reshape(reshape_format[n])
        flat_clips.append(mp.clips_array(collage_clips))

        i += 16

    print('\rconstruction done')
    print('composing final video')

    # compose all segments
    final_clip = mp.concatenate_videoclips(flat_clips[:len(beat_times_bgm) - 1])
    final_clip.write_videofile('chopping.mp4')

    print('composation done')


def audio_extraction():
    # get clip file names
    clip_names = os.listdir(clip_dir)
    # print(clip_names)
    for name in clip_names:
        if not name.endswith('.mp4'):
            continue
        fn = os.path.join(clip_dir, name)
        clip = mp.VideoFileClip(fn)
        clip.audio.write_audiofile(fn[:-4] + '.wav')


if __name__ == '__main__':
    # audio_extraction()
    main('bgm.wav')

    os.system('say "Mission complete."')
