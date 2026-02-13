import pandas as pd
import numpy as np
import joblib 
import librosa
import streamlit as st

st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon=":musical score",
    layout="wide"
)
@st.cache_resource
def load_assets():
    try:
        model=joblib.load('./results/final_ensemble_model.pkl')
        scaler=joblib.load("./results/scaler.pkl")
        le=joblib.load("./results/label_encoder.pkl")
        return model, scaler, le
    except FileNotFoundError:
        return None, None, None
    
model, scaler, le=load_assets()

def compute_features(y,sr):
    features={}

    #basic features (mean and variance)
    chroma=librosa.feature.chroma_stft(y=y,sr=sr)
    features['chroma_stft_mean']=np.mean(chroma)
    features['chroma_stft_var']=np.var(chroma)

    rms=librosa.feature.rms(y=y) #energy
    features['rms_mean'] = np.mean(rms)
    features['rms_var'] = np.var(rms)

    spec_cent=librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spec_cent_mean'] = np.mean(spec_cent)
    features['spec_cent_var'] = np.var(spec_cent)

    spec_bw=librosa.feature.spectral_bandwidth(y=y,sr=sr)
    features['spec_bw_mean'] = np.mean(spec_bw)
    features['spec_bw_var'] = np.var(spec_bw)

    rolloff=librosa.feature.spectral_rolloff(y=y,sr=sr)
    features['rolloff_mean'] = np.mean(rolloff)
    features['rolloff_var'] = np.var(rolloff)

    zcr=librosa.feature.zero_crossing_rate(y=y)
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_var'] = np.var(zcr)

    y_harm, y_perc = librosa.effects.hpss(y=y)
    rms_harm=librosa.feature.rms(y=y_harm)
    features['harmony_mean']=np.mean(rms_harm) #How much the harmonic energy fluctuates.
    features['harmony_var']=np.var(rms_harm)

    rms_perc=librosa.feature.rms(y=y_perc)
    features['perceptr_mean']=np.mean(rms_perc) #Average energy of the drum / transient part of the clip.
    features['perceptr_var']=np.var(rms_perc)
    #advanced features
    #spectral contrast (distinguishes peak-y and noise-like sound)
    contrast=librosa.feature.spectral_contrast(y=y,sr=sr)
    for i in range(contrast.shape[0]):
        features[f'spectral_contrast_mean_{i}']=np.mean(contrast[i])
        features[f'spectral_contrast_var_{i}']=np.var(contrast[i])
    
    #tonnetz (captures harmonic/chord progression - great for jazz and pop)
    try:
        #tonnetz requires harmonic component
        y_harmonic=librosa.effects.harmonic(y)
        tonnetz=librosa.feature.tonnetz(y=y_harmonic,sr=sr)
        for i in range(tonnetz.shape[0]):
            features[f'tonnetz_mean_{i}']=np.mean(tonnetz[i])
            features[f'tonnetz_var_{i}']=np.var(tonnetz[i])
    except:
        #fallback if silence/error
        for i in range(6):features[f'tonnetz_{i}']=0.0

    #mfcc + delta (captures timbre + rhythm/change)
    mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=20)
    for i in range(20):
        features[f'mfcc_mean_{i}']=np.mean(mfcc[i])
        features[f'mfcc_var_{i}']=np.var(mfcc[i])

    mfcc_delta = librosa.feature.delta(mfcc, width=3)
    for i in range(20):
        features[f'delta_mean_{i}']=np.mean(mfcc_delta[i])
        features[f'delta_var_{i}']=np.var(mfcc_delta[i])

    return features

st.title("Music Genre Classfier")
col1=st.columns(1)[0]
with col1:
    st.header("Upload audio")
    uploaded_file=st.file_uploader("Upload a .wav or .mp3 file", type=['wav','mp3'])

    if uploaded_file is not None:
        if model is None:
            st.error("Error! Model not found")
        else:
            st.audio(uploaded_file, format='audio/wav')
            if st.button('Classify genre'):
                with st.spinner("Extracting features and analyzing..."):
                    y,sr=librosa.load(uploaded_file,duration=30)

                    #split into 3-second chunks (voting strategy)
                    chunk_samples=3*sr
                    num_chunks=int(len(y)/chunk_samples)
                    if num_chunks==0:
                        st.error("Audio file too short (minimum 3 seconds required)")
                    chunk_predictions=[]

                    for i in range(num_chunks):
                        start=i*chunk_samples
                        end=start+chunk_samples
                        y_chunk=y[start:end]
                        #extract features
                        feat_dict=compute_features(y_chunk,sr)
                        feat_df=pd.DataFrame([feat_dict])
                        #scaling
                        feat_scaled=scaler.transform(feat_df)
                        #predict
                        pred_idx=model.predict(feat_scaled)[0]
                        chunk_predictions.append(pred_idx)
                    
                    #turn indices into class names
                    pred_labels=le.inverse_transform(chunk_predictions)
                    #find the most frequent label
                    final_prediction=max(set(pred_labels),key=list(pred_labels).count)
                    confidence=list(pred_labels).count(final_prediction)/len(pred_labels)

                    #display the result
                    st.success(f"### Predicted Genre: **{final_prediction.upper()}**")
                    st.info(f"Confidence: {confidence:.0%} (Based on {len(pred_labels)} chunks)")

                    st.write("Chunk Breakdown:")
                    breakdown = pd.Series(pred_labels).value_counts()
                    st.bar_chart(breakdown)