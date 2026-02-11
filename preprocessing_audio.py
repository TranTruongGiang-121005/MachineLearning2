# preprocessing_audio.py
import os
import librosa
import numpy as np

# ============ CONFIG ============
DATASET_PATH = "../genres_original"
OUTPUT_PATH = "../spectrogram_vectors.npz"
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
CLIP_DURATION = 3
FULL_DURATION = 30
# clean output
if os.path.exists(OUTPUT_PATH):
    print(f"\n--- Removing old result file: {OUTPUT_PATH} ---")
    os.remove(OUTPUT_PATH)

# --- Function to split and save spectrogram vectors ---
def convert_audio_into_vectors():
    data=[]
    labels=[]
    song_ids=[]
    print("\n--- Generating spectrogram vectors ---")
    genres=[g for g in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH,g))]

    for genre in genres:
        print(f"Processing genre: {genre}")
        genre_path=os.path.join(DATASET_PATH,genre)
        files=[f for f in os.listdir(genre_path) if f.endswith(".wav")]

        for file in files:
            #jave00054.wav is courrupted
            if file=="jazz.00054.wav":
                print(f"[!] Skipping known corrupt file: {file}")
                continue

            file_path=os.path.join(genre_path,file)
            try:
                #load audio
                y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                #check for empty or too short files
                if len(y) < int(SAMPLE_RATE*1):
                    print(f" Skipped short/empty file: {genre_path}")
                    continue

                # Pad or trim to 30s exactly
                target_len = int(FULL_DURATION * sr)
                y = librosa.util.fix_length(y, size=target_len)

                # 10 clips of 3 seconds each
                samples_per_clip = int(CLIP_DURATION * sr)
                num_segments = target_len // samples_per_clip

                # Create one image per clip
                for i in range(num_segments):
                    start = i * samples_per_clip
                    end = start + samples_per_clip
                    y_clip = y[start:end]

                    # Compute mel-spectrogram
                    S = librosa.feature.melspectrogram(y=y_clip, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
                    S_dB = librosa.power_to_db(S, ref=np.max)

                    #flatten to a 1D vector (128*~130)
                    data.append(S_dB.flatten())
                    labels.append(genre)
                    song_ids.append(file)

            except Exception as e:
                print(f" Error processing {file}: {str(e)}")

    #convert to numpy arrays           
    X=np.array(data)
    y=np.array(labels)
    groups=np.array(song_ids)

    print(f"\n Sucessfully processed {len(np.unique(groups))} songs.")
    print(f"Total segments generated: {len(X)}")

    #save to a compressed file
    print(f"\nSaving {len(X)} vectors to {OUTPUT_PATH}")
    np.savez_compressed(OUTPUT_PATH, X=X, y=y, groups=groups)
    print("Done!")

if __name__ == "__main__":
    convert_audio_into_vectors()
