import librosa
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from scipy.signal import lfilter
import warnings
warnings.filterwarnings('ignore')

# We are removing sklearn and the ML classifier as it was the source of the error.
# We will build a more robust rule-based system based on correct DSP principles.

class VoiceGenderDetector:
    def __init__(self):
        self.sample_rate = 16000
        
        # Gender classification thresholds (rule-based)
        # These are average values and ranges.
        # Male: Lower F0, Lower Formants
        # Female: Higher F0, Higher Formants
        self.male_f0_mean = 130  # Hz
        self.female_f0_mean = 220 # Hz
        
        self.male_f1_mean = 550 # Hz
        self.female_f1_mean = 850 # Hz
        
        self.male_f2_mean = 1500 # Hz
        self.female_f2_mean = 2100 # Hz

    def extract_features(self, audio_data, sr):
        """
        Extract key voice features using robust DSP techniques.
        F0 is found with PYIN.
        Formants (F1, F2) are found using Linear Predictive Coding (LPC).
        """
        if len(audio_data) < 2048: # Need a minimum length for analysis
            return None
        
        features = {}
        
        try:
            # 1. Fundamental Frequency (F0) using a robust algorithm (PYIN)
            f0, voiced_flag, voiced_probs = librosa.pyin(audio_data, 
                                                        fmin=librosa.note_to_hz('C2'), # 65 Hz
                                                        fmax=librosa.note_to_hz('C7'), # 2093 Hz
                                                        sr=sr)
            # Get the mean of only the voiced frames
            f0_clean = f0[~np.isnan(f0)]
            features['f0_mean'] = np.mean(f0_clean) if len(f0_clean) > 0 else 0
            
            # 2. Formants (F1, F2) using Linear Predictive Coding (LPC)
            # This is the CORRECT way to find formants, not FFT peaks.
            
            # Pre-emphasis to boost high frequencies
            pre_emphasis = 0.97
            y_preem = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
            
            # LPC order: rule of thumb is sr/1000 + 2
            lpc_order = int(sr / 1000) + 2
            
            # Get LPC coefficients
            A = librosa.lpc(y_preem, order=lpc_order)
            
            # Get the roots of the LPC polynomial
            rts = np.roots(A)
            
            # Filter out roots that are not on the unit circle
            rts = [r for r in rts if np.imag(r) >= 0]
            
            # Get angles of the roots
            angz = np.angle(rts)
            
            # Convert angles to frequencies (Hz) and sort them
            formants = sorted(angz * (sr / (2 * np.pi)))
            
            # Filter out formants outside the typical human range
            formants = [f for f in formants if 50 < f < 3500]
            
            if len(formants) >= 2:
                features['f1'] = formants[0]
                features['f2'] = formants[1]
            else:
                features['f1'] = 0
                features['f2'] = 0

            # 3. Spectral Centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            features['spectral_centroid'] = np.mean(spectral_centroid)
            
            # 4. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features['zcr'] = np.mean(zcr)
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
        
        return features

    def classify_gender_rules(self, features):
        """
        Robust rule-based gender classification using a scoring system.
        A score close to 0 indicates Male, a score close to 1 indicates Female.
        """
        if features is None or features['f0_mean'] == 0:
            return "Unknown", 0.5
        
        f0 = features['f0_mean']
        f1 = features['f1']
        f2 = features['f2']
        
        scores = []
        
        # Scoring based on F0
        # The closer the F0 is to the typical female F0, the higher the score.
        f0_score = (f0 - self.male_f0_mean) / (self.female_f0_mean - self.male_f0_mean)
        scores.append(np.clip(f0_score, 0, 1)) # Clip score between 0 and 1
        
        # Scoring based on F1
        if f1 > 0:
            f1_score = (f1 - self.male_f1_mean) / (self.female_f1_mean - self.male_f1_mean)
            scores.append(np.clip(f1_score, 0, 1))
        
        # Scoring based on F2
        if f2 > 0:
            f2_score = (f2 - self.male_f2_mean) / (self.female_f2_mean - self.male_f2_mean)
            scores.append(np.clip(f2_score, 0, 1))
            
        if not scores:
            return "Unknown", 0.5
        
        # Final score is the average of all feature scores
        final_score = np.mean(scores)
        
        # Determine confidence based on how far the score is from the middle (0.5)
        confidence = 0.5 + abs(final_score - 0.5)
        
        if final_score > 0.5:
            return "Female", confidence
        else:
            return "Male", confidence

    def process_audio_chunk(self, audio_data, sr):
        """Process a single audio chunk"""
        # Extract features using the new, robust method
        features = self.extract_features(audio_data, sr)
        
        if features is None:
            return "No Voice", 0.5, features
        
        # Classify using our improved rule-based system
        gender, confidence = self.classify_gender_rules(features)
        
        return gender, confidence, features

def analyze_audio_file(audio_file, chunk_duration=2.0):
    """Analyze uploaded audio file"""
    if audio_file is None:
        return None, "Please upload an audio file", None
    
    try:
        detector = VoiceGenderDetector()
        
        # Load audio
        y, sr = librosa.load(audio_file, sr=16000)
        
        # Process in chunks
        chunk_size = int(chunk_duration * sr)
        results = []
        features_history = []
        
        for i in range(0, len(y), chunk_size):
            chunk = y[i:i+chunk_size]
            if len(chunk) < sr * 0.5:  # Skip very short chunks at the end
                continue
            
            gender, confidence, features = detector.process_audio_chunk(chunk, sr)
            timestamp = i / sr
            
            results.append({
                'time': timestamp,
                'gender': gender,
                'confidence': confidence,
            })
            
            if features:
                # Add timestamp to features for plotting
                features['time'] = timestamp
                features_history.append(features)
        
        if not results:
            return None, "Could not process the audio. It might be too short or silent.", None

        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Voice Gender Analysis', fontsize=16)
        
        # Plot 1: Waveform
        librosa.display.waveshow(y, sr=sr, ax=ax1, alpha=0.7)
        ax1.set_title('Audio Waveform')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Gender Detection Over Time
        times = [r['time'] for r in results]
        genders = [r['gender'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        colors = ['blue' if g == 'Male' else 'red' if g == 'Female' else 'gray' for g in genders]
        
        ax2.scatter(times, confidences, c=colors, alpha=0.7, s=50, label=genders)
        ax2.set_title('Gender Detection Confidence Over Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Confidence')
        ax2.set_ylim(0.4, 1.05) # Start at 0.4 to better see confidence variations
        ax2.grid(True, alpha=0.3)
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Male', markerfacecolor='blue', markersize=10),
                           Line2D([0], [0], marker='o', color='w', label='Female', markerfacecolor='red', markersize=10),
                           Line2D([0], [0], marker='o', color='w', label='Unknown', markerfacecolor='gray', markersize=10)]
        ax2.legend(handles=legend_elements, loc='best')

        # Plot 3: Fundamental Frequency (F0) and Formants (F1)
        if features_history:
            f_times = [f['time'] for f in features_history]
            f0_values = [f['f0_mean'] for f in features_history if f['f0_mean'] > 0]
            f0_times = [f['time'] for f in features_history if f['f0_mean'] > 0]
            f1_values = [f['f1'] for f in features_history if f.get('f1', 0) > 0]
            f1_times = [f['time'] for f in features_history if f.get('f1', 0) > 0]

            ax3.plot(f0_times, f0_values, 'o-', color='green', linewidth=2, label='F0 (Pitch)')
            ax3.plot(f1_times, f1_values, 's--', color='purple', linewidth=2, alpha=0.7, label='F1 (Formant 1)')

            ax3.set_title('Fundamental Frequency (F0) & First Formant (F1)')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Frequency (Hz)')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            ax3.set_yscale('log') # Log scale is often better for pitch
            ax3.set_ylim(bottom=70, top=1200)

        # Plot 4: Spectral Centroid
        if features_history:
            sc_values = [f['spectral_centroid'] for f in features_history]
            ax4.plot(times, sc_values, 'o-', color='orange', linewidth=2)
            ax4.set_title('Spectral Centroid (Brightness)')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Frequency (Hz)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Generate summary statistics
        male_count = sum(1 for r in results if r['gender'] == 'Male')
        female_count = sum(1 for r in results if r['gender'] == 'Female')
        
        total_chunks = len(results)
        
        # Overall classification based on majority vote
        if male_count > female_count:
            overall_gender = "Male"
            overall_confidence = (male_count / total_chunks) * 100
        elif female_count > male_count:
            overall_gender = "Female"
            overall_confidence = (female_count / total_chunks) * 100
        else:
            overall_gender = "Indeterminate"
            overall_confidence = 50

        summary = f"""
        ## 🎤 Voice Gender Analysis Results
        
        **Overall Classification:** **{overall_gender}** (Based on {overall_confidence:.1f}% of analyzed chunks)
        
        ### 📊 Detailed Statistics:
        - **Total Chunks Analyzed:** {total_chunks}
        - **Male Chunks:** {male_count} ({male_count/total_chunks:.1%})
        - **Female Chunks:** {female_count} ({female_count/total_chunks:.1%})
        
        ### 🔊 Audio Features (Averages):
        - **Duration:** {len(y)/sr:.2f} seconds
        """
        
        if features_history:
            avg_f0 = np.mean([f['f0_mean'] for f in features_history if f['f0_mean'] > 0])
            avg_f1 = np.mean([f['f1'] for f in features_history if f.get('f1', 0) > 0])
            avg_spectral_centroid = np.mean([f['spectral_centroid'] for f in features_history])
            
            summary += f"""
            - **Average F0 (Pitch):** {avg_f0:.1f} Hz
            - **Average F1 (Formant 1):** {avg_f1:.1f} Hz
            - **Average Spectral Centroid:** {avg_spectral_centroid:.1f} Hz
            """
        
        return fig, summary
        
    except Exception as e:
        return None, f"Error analyzing audio: {str(e)}"

def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(title="Voice Gender Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎤 Voice Gender Detection System (DSP Project)
        
        This system uses Digital Signal Processing (DSP) techniques to analyze voice characteristics 
        and classify gender. It **does not use Machine Learning**, but instead relies on a robust scoring system based on acoustic phonetics.
        
        ## 🧠 How it Works:
        - **Fundamental Frequency (F0)**: Extracts the speaker's pitch using the PYIN algorithm.
        - **Formants (F1, F2)**: Estimates the vocal tract resonances using **Linear Predictive Coding (LPC)**.
        - **Scoring System**: Each feature (F0, F1, F2) is scored based on how close it is to typical male vs. female frequency ranges. The final classification is an average of these scores.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Audio Input")
                audio_input = gr.Audio(
                    label="Upload Audio File (WAV, MP3, etc.)",
                    type="filepath"
                )
                
                gr.Markdown("### ⚙️ Analysis Parameters")
                chunk_duration = gr.Slider(
                    minimum=0.5, maximum=4.0, value=1.5, step=0.5,
                    label="Chunk Duration (seconds)"
                )
                
                analyze_btn = gr.Button("🔍 Analyze Voice Gender", variant="primary")
                
            with gr.Column(scale=2):
                gr.Markdown("### 📈 Analysis Results")
                plot_output = gr.Plot(label="Voice Analysis Visualization")
                summary_output = gr.Markdown()
        
        # Set up the analysis function
        analyze_btn.click(
            fn=analyze_audio_file,
            inputs=[audio_input, chunk_duration],
            outputs=[plot_output, summary_output]
        )
        
        # Add technical information
        gr.Markdown("""
        ## 🔬 Technical Details
        
        ### Features Extracted per Chunk:
        1. **Fundamental Frequency (F0)**: The primary pitch, a strong indicator of gender.
        2. **Formants (F1, F2)**: Key resonances of the vocal tract, which differ in size between sexes.
        3. **Spectral Centroid**: The "brightness" of the sound.
        
        ### Classification Method:
        - A **rule-based scoring engine** calculates a value between 0 ( prototypically Male) and 1 (prototypically Female) for each feature.
        - The final decision is based on the average score across all features, making it robust to variations in a single feature.
        
        ### Known Gender Differences in Speech:
        - **Male voices**: Lower F0 (~85-180 Hz), lower formants (longer vocal tract).
        - **Female voices**: Higher F0 (~165-265 Hz), higher formants (shorter vocal tract).
        """)
    
    return demo

# Run the application
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)