"""
ASR Demo for Disordered Speech with Error Analysis

A Streamlit-based web application for demonstrating Automatic Speech Recognition
capabilities and analyzing transcription errors, particularly for disordered speech.

Usage:
    streamlit run app.py

Author: ASR-Error Project
License: MIT
"""

import json
import tempfile
import os
from dataclasses import asdict

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.preprocess import preprocess_audio, detect_pauses, AudioMetadata, PauseInfo
from src.asr_utils import ASRTranscriber, TranscriptionResult, get_available_models, DEFAULT_MODEL
from src.analysis_utils import (
    analyze_transcription,
    format_analysis_for_export,
    ErrorType,
    AnalysisResult,
)


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="ASR Demo - Disordered Speech Analysis",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS
# ============================================================================

st.markdown("""
<style>
    /* Main theme */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Fira Code', 'JetBrains Mono', monospace;
        color: #e94560;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1f4068, #162447);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #e94560;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #e94560;
        font-family: 'Fira Code', monospace;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Diff highlighting */
    .diff-match {
        background-color: rgba(0, 200, 83, 0.2);
        color: #00c853;
        padding: 2px 6px;
        border-radius: 4px;
        margin: 2px;
        display: inline-block;
    }
    
    .diff-substitution {
        background-color: rgba(255, 82, 82, 0.2);
        color: #ff5252;
        padding: 2px 6px;
        border-radius: 4px;
        margin: 2px;
        display: inline-block;
        text-decoration: underline;
    }
    
    .diff-insertion {
        background-color: rgba(255, 193, 7, 0.2);
        color: #ffc107;
        padding: 2px 6px;
        border-radius: 4px;
        margin: 2px;
        display: inline-block;
        font-style: italic;
    }
    
    .diff-deletion {
        background-color: rgba(156, 39, 176, 0.2);
        color: #9c27b0;
        padding: 2px 6px;
        border-radius: 4px;
        margin: 2px;
        display: inline-block;
        text-decoration: line-through;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(145deg, #2d3a4f, #1e2838);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #3d5a80;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(145deg, #e94560, #c73e54);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(145deg, #ff6b81, #e94560);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(233, 69, 96, 0.4);
    }
    
    /* Pattern cards */
    .pattern-card {
        background: linear-gradient(145deg, #2d3a4f, #1e2838);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 3px solid #ffc107;
    }
    
    /* Warning box */
    .warning-box {
        background: linear-gradient(145deg, #4a3f35, #3a2f25);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

if 'transcriber' not in st.session_state:
    st.session_state.transcriber = None
if 'transcription_result' not in st.session_state:
    st.session_state.transcription_result = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'audio_metadata' not in st.session_state:
    st.session_state.audio_metadata = None
if 'pause_info' not in st.session_state:
    st.session_state.pause_info = None


# ============================================================================
# Helper Functions
# ============================================================================

def render_diff_html(analysis: AnalysisResult) -> str:
    """Generate HTML for color-coded diff visualization."""
    html_parts = []
    
    for token in analysis.diff_tokens:
        if token.error_type == ErrorType.MATCH:
            html_parts.append(f'<span class="diff-match">{token.text}</span>')
        elif token.error_type == ErrorType.SUBSTITUTION:
            html_parts.append(
                f'<span class="diff-substitution" title="Expected: {token.reference}">'
                f'{token.text}</span>'
            )
        elif token.error_type == ErrorType.INSERTION:
            html_parts.append(
                f'<span class="diff-insertion" title="Extra word">{token.text}</span>'
            )
        elif token.error_type == ErrorType.DELETION:
            html_parts.append(
                f'<span class="diff-deletion" title="Missing word">{token.text}</span>'
            )
    
    return ' '.join(html_parts)


def create_error_distribution_chart(analysis: AnalysisResult) -> plt.Figure:
    """Create a pie chart showing error distribution."""
    breakdown = analysis.error_breakdown
    
    labels = ['Correct', 'Substitutions', 'Insertions', 'Deletions']
    values = [
        breakdown.correct,
        breakdown.substitutions,
        breakdown.insertions,
        breakdown.deletions
    ]
    colors = ['#00c853', '#ff5252', '#ffc107', '#9c27b0']
    
    # Filter out zero values
    non_zero = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if not non_zero:
        return None
    
    labels, values, colors = zip(*non_zero)
    
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=[0.02] * len(values),
        shadow=True
    )
    
    # Style the text
    for text in texts:
        text.set_color('#e0e0e0')
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_color('#1a1a2e')
        autotext.set_fontweight('bold')
    
    ax.set_title('Error Distribution', color='#e0e0e0', fontsize=14, pad=20)
    
    return fig


def create_metrics_bar_chart(analysis: AnalysisResult) -> plt.Figure:
    """Create a bar chart showing error rates."""
    breakdown = analysis.error_breakdown
    
    metrics = {
        'WER': analysis.wer * 100,
        'CER': analysis.cer * 100,
        'Sub Rate': breakdown.substitution_rate * 100,
        'Ins Rate': breakdown.insertion_rate * 100,
        'Del Rate': breakdown.deletion_rate * 100,
    }
    
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    bars = ax.bar(
        metrics.keys(),
        metrics.values(),
        color=['#e94560', '#ff6b81', '#ff5252', '#ffc107', '#9c27b0'],
        edgecolor='#e0e0e0',
        linewidth=0.5
    )
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics.values()):
        height = bar.get_height()
        ax.annotate(
            f'{value:.1f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            color='#e0e0e0',
            fontsize=9
        )
    
    ax.set_ylabel('Percentage (%)', color='#e0e0e0')
    ax.set_ylim(0, max(metrics.values()) * 1.2 if max(metrics.values()) > 0 else 10)
    ax.tick_params(colors='#e0e0e0')
    ax.spines['bottom'].set_color('#e0e0e0')
    ax.spines['left'].set_color('#e0e0e0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


@st.cache_resource
def load_transcriber(model_name: str) -> ASRTranscriber:
    """Load and cache the ASR transcriber."""
    return ASRTranscriber(model_name=model_name)


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Model selection
    st.markdown("### 🤖 ASR Model")
    available_models = get_available_models()
    selected_model = st.selectbox(
        "Select Model",
        available_models,
        index=available_models.index(DEFAULT_MODEL),
        help="Choose the ASR model for transcription"
    )
    
    st.markdown("---")
    
    # Info section
    st.markdown("### ℹ️ About")
    st.markdown("""
    This tool demonstrates ASR capabilities for analyzing 
    disordered speech patterns.
    
    **Features:**
    - 🎵 Audio upload & preprocessing
    - 📝 Automatic transcription
    - 📊 Error analysis (WER/CER)
    - 🔍 Pattern detection
    - 📥 JSON export
    """)
    
    st.markdown("---")
    
    # Warning
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Disclaimer</strong><br>
        This tool is for demonstration purposes only. 
        It should not be used for clinical diagnosis.
        Always consult qualified professionals for 
        speech pathology assessment.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Legend
    st.markdown("### 🎨 Color Legend")
    st.markdown("""
    <span class="diff-match">Match</span> - Correct words<br>
    <span class="diff-substitution">Substitution</span> - Wrong word<br>
    <span class="diff-insertion">Insertion</span> - Extra word<br>
    <span class="diff-deletion">Deletion</span> - Missing word
    """, unsafe_allow_html=True)


# ============================================================================
# Main Content
# ============================================================================

st.markdown("# 🎙️ ASR Demo for Disordered Speech")
st.markdown("### Automatic Speech Recognition with Error Analysis")

st.markdown("---")

# Two-column layout for input
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📁 Upload Audio")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3'],
        help="Upload a WAV or MP3 file (max 10MB)"
    )
    
    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')
        st.success(f"✅ File loaded: {uploaded_file.name}")

with col2:
    st.markdown("### 📝 Ground Truth")
    ground_truth = st.text_area(
        "Enter the reference transcription",
        height=150,
        placeholder="Type or paste the correct transcription here...",
        help="This is the expected/correct text for comparison"
    )
    
    # Option to upload ground truth from file
    gt_file = st.file_uploader(
        "Or upload a text file",
        type=['txt'],
        help="Upload a .txt file containing the ground truth"
    )
    
    if gt_file:
        ground_truth = gt_file.read().decode('utf-8')
        st.text_area("Loaded ground truth:", ground_truth, height=100, disabled=True)

st.markdown("---")

# Process button
if st.button("🚀 Analyze Speech", use_container_width=True):
    if not uploaded_file:
        st.error("⚠️ Please upload an audio file first!")
    elif not ground_truth.strip():
        st.error("⚠️ Please provide the ground truth text!")
    else:
        with st.spinner("🔄 Processing audio..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                # Preprocess audio
                audio_array, metadata = preprocess_audio(tmp_path)
                st.session_state.audio_metadata = metadata
                
                # Detect pauses
                pause_info = detect_pauses(audio_array, metadata.sample_rate)
                st.session_state.pause_info = pause_info
                
                # Load transcriber
                transcriber = load_transcriber(selected_model)
                
                # Transcribe
                with st.spinner("🎯 Transcribing audio..."):
                    result = transcriber.transcribe(audio_array)
                    st.session_state.transcription_result = result
                
                # Analyze
                with st.spinner("📊 Analyzing errors..."):
                    analysis = analyze_transcription(
                        ground_truth,
                        result.text,
                        audio_duration_seconds=metadata.duration_seconds,
                        pause_info={
                            'num_pauses': pause_info.num_pauses,
                            'total_pause_duration': pause_info.total_pause_duration
                        }
                    )
                    st.session_state.analysis_result = analysis
                
                # Cleanup
                os.unlink(tmp_path)
                
                st.success("✅ Analysis complete!")
                
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                raise e

# ============================================================================
# Results Display
# ============================================================================

if st.session_state.analysis_result:
    analysis = st.session_state.analysis_result
    transcription = st.session_state.transcription_result
    metadata = st.session_state.audio_metadata
    pause_info = st.session_state.pause_info
    
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # Metrics row
    metric_cols = st.columns(5)
    
    with metric_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{analysis.wer * 100:.1f}%</div>
            <div class="metric-label">Word Error Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{analysis.cer * 100:.1f}%</div>
            <div class="metric-label">Character Error Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols[2]:
        speech_rate = analysis.speech_metrics.speech_rate_wpm if analysis.speech_metrics else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{speech_rate:.0f}</div>
            <div class="metric-label">Words per Minute</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols[3]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{pause_info.num_pauses}</div>
            <div class="metric-label">Detected Pauses</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols[4]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metadata.duration_seconds:.1f}s</div>
            <div class="metric-label">Audio Duration</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Transcription comparison
    st.markdown("### 📝 Transcription Comparison")
    
    comp_cols = st.columns(2)
    
    with comp_cols[0]:
        st.markdown("**Ground Truth:**")
        st.markdown(f"""
        <div class="info-box">
            {ground_truth}
        </div>
        """, unsafe_allow_html=True)
    
    with comp_cols[1]:
        st.markdown("**ASR Transcription:**")
        st.markdown(f"""
        <div class="info-box">
            {transcription.text}
        </div>
        """, unsafe_allow_html=True)
    
    # Diff visualization
    st.markdown("### 🔍 Error Highlighting")
    st.markdown("""
    <div class="info-box">
        {}
    </div>
    """.format(render_diff_html(analysis)), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    st.markdown("### 📈 Visualizations")
    
    chart_cols = st.columns(2)
    
    with chart_cols[0]:
        fig = create_error_distribution_chart(analysis)
        if fig:
            st.pyplot(fig)
            plt.close(fig)
    
    with chart_cols[1]:
        fig = create_metrics_bar_chart(analysis)
        st.pyplot(fig)
        plt.close(fig)
    
    # Error breakdown table
    st.markdown("### 📋 Error Breakdown")
    
    breakdown_data = {
        'Metric': ['Substitutions', 'Insertions', 'Deletions', 'Correct', 'Total Words'],
        'Count': [
            analysis.error_breakdown.substitutions,
            analysis.error_breakdown.insertions,
            analysis.error_breakdown.deletions,
            analysis.error_breakdown.correct,
            analysis.error_breakdown.total_words
        ],
        'Rate': [
            f"{analysis.error_breakdown.substitution_rate * 100:.1f}%",
            f"{analysis.error_breakdown.insertion_rate * 100:.1f}%",
            f"{analysis.error_breakdown.deletion_rate * 100:.1f}%",
            f"{(analysis.error_breakdown.correct / analysis.error_breakdown.total_words * 100) if analysis.error_breakdown.total_words > 0 else 0:.1f}%",
            "100%"
        ]
    }
    
    st.dataframe(
        pd.DataFrame(breakdown_data),
        use_container_width=True,
        hide_index=True
    )
    
    # Phonetic patterns
    if analysis.phonetic_patterns:
        st.markdown("---")
        st.markdown("### 🧬 Detected Phonetic Patterns")
        st.markdown("""
        <div class="warning-box">
            <strong>Note:</strong> These patterns are identified through simple rule-based analysis 
            and should be interpreted with caution. They are not diagnostic.
        </div>
        """, unsafe_allow_html=True)
        
        for pattern in analysis.phonetic_patterns:
            st.markdown(f"""
            <div class="pattern-card">
                <h4 style="color: #ffc107; margin: 0;">🔸 {pattern.pattern_name}</h4>
                <p style="color: #b0b0b0; margin: 5px 0;">{pattern.description}</p>
                <p style="color: #a0a0a0; font-size: 0.9em;">
                    <strong>Frequency:</strong> {pattern.frequency} occurrences<br>
                    <strong>Clinical Note:</strong> {pattern.clinical_significance}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Export section
    st.markdown("---")
    st.markdown("### 📥 Export Results")
    
    export_data = format_analysis_for_export(analysis)
    export_data['transcription'] = transcription.text
    export_data['ground_truth'] = ground_truth
    export_data['model_used'] = transcription.model_name
    export_data['processing_time_seconds'] = transcription.processing_time_seconds
    
    export_json = json.dumps(export_data, indent=2)
    
    st.download_button(
        label="📥 Download JSON Report",
        data=export_json,
        file_name="asr_analysis_results.json",
        mime="application/json",
        use_container_width=True
    )
    
    with st.expander("🔎 Preview JSON"):
        st.json(export_data)


# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #707070; padding: 20px;">
    <p>ASR Demo for Disordered Speech Analysis</p>
    <p style="font-size: 0.8em;">
        Built with Streamlit • Powered by Hugging Face Transformers<br>
        For research and educational purposes only
    </p>
    <p style="font-size: 0.8em; margin-top: 10px; color: #909090;">
        Developed by Ahmet Yildiz - using Cursor
    </p>
</div>
""", unsafe_allow_html=True)
