"""
Error Analysis and Metrics Module for ASR Demo

This module provides functionality for analyzing transcription errors,
computing metrics like Word Error Rate (WER), and identifying patterns
that may be indicative of speech disorders.

Key Features:
- WER calculation with detailed error breakdown
- Speech rate computation
- Asymmetric error detection for disordered speech
- Color-coded diff generation for visualization

References:
- JiWER Library: https://github.com/jitsi/jiwer
- Word Error Rate: https://en.wikipedia.org/wiki/Word_error_rate
- Speech Pathology Indicators: Yorkston et al. (1996) "Assessment of Motor Speech Disorders"
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

from jiwer import wer, cer, process_words


# ============================================================================
# Data Classes
# ============================================================================

class ErrorType(Enum):
    """Types of transcription errors."""
    SUBSTITUTION = "substitution"
    INSERTION = "insertion"
    DELETION = "deletion"
    MATCH = "match"


@dataclass
class WordError:
    """Represents a single word-level error."""
    error_type: ErrorType
    reference_word: Optional[str]  # Word from ground truth
    hypothesis_word: Optional[str]  # Word from transcription
    position: int  # Position in the aligned sequence


@dataclass
class ErrorBreakdown:
    """Detailed breakdown of transcription errors."""
    substitutions: int
    insertions: int
    deletions: int
    correct: int
    total_words: int  # Total words in reference
    
    @property
    def total_errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions
    
    @property
    def substitution_rate(self) -> float:
        return self.substitutions / self.total_words if self.total_words > 0 else 0.0
    
    @property
    def insertion_rate(self) -> float:
        return self.insertions / self.total_words if self.total_words > 0 else 0.0
    
    @property
    def deletion_rate(self) -> float:
        return self.deletions / self.total_words if self.total_words > 0 else 0.0


@dataclass
class DiffToken:
    """A token in the diff visualization."""
    text: str
    error_type: ErrorType
    reference: Optional[str] = None  # For substitutions


@dataclass 
class SpeechMetrics:
    """Speech-related metrics computed from audio and transcription."""
    speech_rate_wpm: float  # Words per minute
    total_words: int
    audio_duration_seconds: float
    pause_count: Optional[int] = None
    total_pause_duration: Optional[float] = None
    average_pause_duration: Optional[float] = None


@dataclass
class PhoneticPattern:
    """Pattern of phonetic errors that may indicate speech disorders."""
    pattern_name: str
    description: str
    affected_words: List[Tuple[str, str]]  # (reference, hypothesis) pairs
    frequency: int
    clinical_significance: str


@dataclass
class AnalysisResult:
    """Complete analysis result combining all metrics."""
    wer: float
    cer: float
    error_breakdown: ErrorBreakdown
    word_errors: List[WordError]
    diff_tokens: List[DiffToken]
    speech_metrics: Optional[SpeechMetrics] = None
    phonetic_patterns: List[PhoneticPattern] = field(default_factory=list)


# ============================================================================
# Core Analysis Functions
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    
    - Convert to lowercase
    - Remove punctuation
    - Normalize whitespace
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation (keep apostrophes for contractions)
    text = re.sub(r"[^\w\s']", "", text)
    
    # Normalize whitespace
    text = " ".join(text.split())
    
    return text


def compute_wer(ground_truth: str, transcription: str) -> float:
    """
    Compute Word Error Rate between ground truth and transcription.
    
    WER = (S + D + I) / N
    where:
    - S = Substitutions
    - D = Deletions
    - I = Insertions
    - N = Total words in reference
    
    Args:
        ground_truth: Reference text
        transcription: ASR output text
        
    Returns:
        Word Error Rate as a float (0.0 = perfect, 1.0 = 100% error)
    """
    gt_normalized = normalize_text(ground_truth)
    trans_normalized = normalize_text(transcription)
    
    if not gt_normalized:
        return 1.0 if trans_normalized else 0.0
    
    return wer(gt_normalized, trans_normalized)


def compute_cer(ground_truth: str, transcription: str) -> float:
    """
    Compute Character Error Rate between ground truth and transcription.
    
    Args:
        ground_truth: Reference text
        transcription: ASR output text
        
    Returns:
        Character Error Rate as a float
    """
    gt_normalized = normalize_text(ground_truth)
    trans_normalized = normalize_text(transcription)
    
    if not gt_normalized:
        return 1.0 if trans_normalized else 0.0
    
    return cer(gt_normalized, trans_normalized)


def get_error_breakdown(ground_truth: str, transcription: str) -> ErrorBreakdown:
    """
    Get detailed breakdown of errors.
    
    Args:
        ground_truth: Reference text
        transcription: ASR output text
        
    Returns:
        ErrorBreakdown with counts of each error type
    """
    gt_normalized = normalize_text(ground_truth)
    trans_normalized = normalize_text(transcription)
    
    if not gt_normalized:
        # Edge case: empty ground truth
        trans_words = trans_normalized.split() if trans_normalized else []
        return ErrorBreakdown(
            substitutions=0,
            insertions=len(trans_words),
            deletions=0,
            correct=0,
            total_words=0
        )
    
    # Use process_words to get detailed error counts
    output = process_words(gt_normalized, trans_normalized)
    
    return ErrorBreakdown(
        substitutions=output.substitutions,
        insertions=output.insertions,
        deletions=output.deletions,
        correct=output.hits,
        total_words=len(gt_normalized.split())
    )


def get_word_errors(ground_truth: str, transcription: str) -> List[WordError]:
    """
    Get list of individual word-level errors with alignment.
    
    Args:
        ground_truth: Reference text
        transcription: ASR output text
        
    Returns:
        List of WordError objects describing each error
    """
    gt_normalized = normalize_text(ground_truth)
    trans_normalized = normalize_text(transcription)
    
    if not gt_normalized and not trans_normalized:
        return []
    
    # Use jiwer's word processing for alignment
    output = process_words(gt_normalized, trans_normalized)
    
    errors = []
    position = 0
    
    # Process alignments from jiwer
    for chunk in output.alignments[0]:
        ref_start, ref_end = chunk.ref_start_idx, chunk.ref_end_idx
        hyp_start, hyp_end = chunk.hyp_start_idx, chunk.hyp_end_idx
        
        ref_words = output.references[0][ref_start:ref_end]
        hyp_words = output.hypotheses[0][hyp_start:hyp_end]
        
        if chunk.type == 'equal':
            for ref, hyp in zip(ref_words, hyp_words):
                errors.append(WordError(
                    error_type=ErrorType.MATCH,
                    reference_word=ref,
                    hypothesis_word=hyp,
                    position=position
                ))
                position += 1
        elif chunk.type == 'substitute':
            for ref, hyp in zip(ref_words, hyp_words):
                errors.append(WordError(
                    error_type=ErrorType.SUBSTITUTION,
                    reference_word=ref,
                    hypothesis_word=hyp,
                    position=position
                ))
                position += 1
        elif chunk.type == 'delete':
            for ref in ref_words:
                errors.append(WordError(
                    error_type=ErrorType.DELETION,
                    reference_word=ref,
                    hypothesis_word=None,
                    position=position
                ))
                position += 1
        elif chunk.type == 'insert':
            for hyp in hyp_words:
                errors.append(WordError(
                    error_type=ErrorType.INSERTION,
                    reference_word=None,
                    hypothesis_word=hyp,
                    position=position
                ))
                position += 1
    
    return errors


def generate_diff_tokens(ground_truth: str, transcription: str) -> List[DiffToken]:
    """
    Generate diff tokens for visualization.
    
    Creates a list of tokens that can be rendered with color coding:
    - Green: Correct matches
    - Red: Substitutions (shows both reference and hypothesis)
    - Yellow: Insertions (extra words in transcription)
    - Gray/Strikethrough: Deletions (missing words from reference)
    
    Args:
        ground_truth: Reference text
        transcription: ASR output text
        
    Returns:
        List of DiffToken objects for visualization
    """
    word_errors = get_word_errors(ground_truth, transcription)
    
    tokens = []
    for error in word_errors:
        if error.error_type == ErrorType.MATCH:
            tokens.append(DiffToken(
                text=error.hypothesis_word,
                error_type=ErrorType.MATCH
            ))
        elif error.error_type == ErrorType.SUBSTITUTION:
            tokens.append(DiffToken(
                text=error.hypothesis_word,
                error_type=ErrorType.SUBSTITUTION,
                reference=error.reference_word
            ))
        elif error.error_type == ErrorType.INSERTION:
            tokens.append(DiffToken(
                text=error.hypothesis_word,
                error_type=ErrorType.INSERTION
            ))
        elif error.error_type == ErrorType.DELETION:
            tokens.append(DiffToken(
                text=error.reference_word,
                error_type=ErrorType.DELETION
            ))
    
    return tokens


# ============================================================================
# Speech Metrics Functions
# ============================================================================

def compute_speech_rate(
    transcription: str,
    duration_seconds: float
) -> SpeechMetrics:
    """
    Compute speech rate and related metrics.
    
    Args:
        transcription: Transcribed text
        duration_seconds: Audio duration in seconds
        
    Returns:
        SpeechMetrics object with speech rate information
    """
    words = normalize_text(transcription).split()
    total_words = len(words)
    
    # Calculate words per minute
    if duration_seconds > 0:
        speech_rate_wpm = (total_words / duration_seconds) * 60
    else:
        speech_rate_wpm = 0.0
    
    return SpeechMetrics(
        speech_rate_wpm=speech_rate_wpm,
        total_words=total_words,
        audio_duration_seconds=duration_seconds
    )


def compute_speech_rate_with_pauses(
    transcription: str,
    duration_seconds: float,
    pause_count: int,
    total_pause_duration: float
) -> SpeechMetrics:
    """
    Compute speech rate including pause analysis.
    
    This is useful for analyzing disordered speech where
    pause patterns may indicate specific conditions.
    
    Args:
        transcription: Transcribed text
        duration_seconds: Total audio duration
        pause_count: Number of detected pauses
        total_pause_duration: Total duration of pauses
        
    Returns:
        SpeechMetrics with pause information included
    """
    metrics = compute_speech_rate(transcription, duration_seconds)
    
    metrics.pause_count = pause_count
    metrics.total_pause_duration = total_pause_duration
    metrics.average_pause_duration = (
        total_pause_duration / pause_count if pause_count > 0 else 0.0
    )
    
    return metrics


# ============================================================================
# Phonetic Pattern Analysis (for Disordered Speech)
# ============================================================================

# Common phonetic patterns associated with speech disorders
CONSONANT_CLASSES = {
    'fricatives': ['f', 'v', 's', 'z', 'sh', 'th'],
    'stops': ['p', 'b', 't', 'd', 'k', 'g'],
    'nasals': ['m', 'n', 'ng'],
    'liquids': ['l', 'r'],
    'glides': ['w', 'y'],
}


def identify_phonetic_patterns(
    word_errors: List[WordError]
) -> List[PhoneticPattern]:
    """
    Identify phonetic error patterns that may indicate speech disorders.
    
    This performs simple rule-based analysis to identify:
    - Higher errors in fricatives (common in Dysarthria)
    - Consonant cluster reduction
    - Final consonant deletion
    - Vowel distortions
    
    Note: This is a simplified analysis. Clinical assessment requires
    professional evaluation.
    
    Args:
        word_errors: List of word-level errors
        
    Returns:
        List of identified phonetic patterns
    """
    patterns = []
    
    # Collect substitution pairs
    substitutions = [
        (e.reference_word, e.hypothesis_word)
        for e in word_errors
        if e.error_type == ErrorType.SUBSTITUTION
        and e.reference_word and e.hypothesis_word
    ]
    
    deletions = [
        e.reference_word
        for e in word_errors
        if e.error_type == ErrorType.DELETION
        and e.reference_word
    ]
    
    # Pattern 1: Fricative errors (common in Dysarthria)
    fricative_errors = []
    for ref, hyp in substitutions:
        for f in CONSONANT_CLASSES['fricatives']:
            if f in ref.lower() and f not in hyp.lower():
                fricative_errors.append((ref, hyp))
                break
    
    if fricative_errors:
        patterns.append(PhoneticPattern(
            pattern_name="Fricative Substitution",
            description="Fricative sounds (f, v, s, z, sh, th) are being substituted or omitted",
            affected_words=fricative_errors,
            frequency=len(fricative_errors),
            clinical_significance="Common in Dysarthria, may indicate reduced articulatory precision"
        ))
    
    # Pattern 2: Final consonant deletion
    final_consonant_deletions = []
    for ref in deletions:
        if len(ref) > 1 and ref[-1].isalpha() and ref[-1] not in 'aeiou':
            final_consonant_deletions.append((ref, "[deleted]"))
    
    for ref, hyp in substitutions:
        if (len(ref) > len(hyp) and 
            ref[:-1].lower() == hyp.lower() and 
            ref[-1] not in 'aeiou'):
            final_consonant_deletions.append((ref, hyp))
    
    if final_consonant_deletions:
        patterns.append(PhoneticPattern(
            pattern_name="Final Consonant Deletion",
            description="Consonants at the end of words are being omitted",
            affected_words=final_consonant_deletions,
            frequency=len(final_consonant_deletions),
            clinical_significance="May indicate phonological processing difficulties"
        ))
    
    # Pattern 3: Stop consonant errors
    stop_errors = []
    for ref, hyp in substitutions:
        for s in CONSONANT_CLASSES['stops']:
            if s in ref.lower() and s not in hyp.lower():
                stop_errors.append((ref, hyp))
                break
    
    if stop_errors:
        patterns.append(PhoneticPattern(
            pattern_name="Stop Consonant Errors",
            description="Stop consonants (p, b, t, d, k, g) showing errors",
            affected_words=stop_errors,
            frequency=len(stop_errors),
            clinical_significance="May indicate reduced motor control for plosive sounds"
        ))
    
    # Pattern 4: Word length effect (longer words have more errors)
    long_word_errors = [
        (ref, hyp) for ref, hyp in substitutions
        if len(ref) > 6
    ]
    
    if len(long_word_errors) > 2:
        patterns.append(PhoneticPattern(
            pattern_name="Word Length Effect",
            description="Increased errors in longer/complex words",
            affected_words=long_word_errors,
            frequency=len(long_word_errors),
            clinical_significance="May indicate motor planning difficulties (apraxia) or fatigue"
        ))
    
    return patterns


# ============================================================================
# Complete Analysis Function
# ============================================================================

def analyze_transcription(
    ground_truth: str,
    transcription: str,
    audio_duration_seconds: Optional[float] = None,
    pause_info: Optional[Dict[str, Any]] = None
) -> AnalysisResult:
    """
    Perform complete analysis of transcription against ground truth.
    
    This is the main entry point for error analysis, combining:
    - WER/CER calculation
    - Error breakdown
    - Word-level error detection
    - Diff generation
    - Speech metrics (if duration provided)
    - Phonetic pattern analysis
    
    Args:
        ground_truth: Reference text
        transcription: ASR output text
        audio_duration_seconds: Optional audio duration for speech rate
        pause_info: Optional pause analysis info (from preprocess module)
        
    Returns:
        AnalysisResult with all analysis data
    """
    # Core metrics
    wer_score = compute_wer(ground_truth, transcription)
    cer_score = compute_cer(ground_truth, transcription)
    error_breakdown = get_error_breakdown(ground_truth, transcription)
    word_errors = get_word_errors(ground_truth, transcription)
    diff_tokens = generate_diff_tokens(ground_truth, transcription)
    
    # Speech metrics (if duration provided)
    speech_metrics = None
    if audio_duration_seconds is not None:
        if pause_info:
            speech_metrics = compute_speech_rate_with_pauses(
                transcription,
                audio_duration_seconds,
                pause_info.get('num_pauses', 0),
                pause_info.get('total_pause_duration', 0.0)
            )
        else:
            speech_metrics = compute_speech_rate(transcription, audio_duration_seconds)
    
    # Phonetic pattern analysis
    phonetic_patterns = identify_phonetic_patterns(word_errors)
    
    return AnalysisResult(
        wer=wer_score,
        cer=cer_score,
        error_breakdown=error_breakdown,
        word_errors=word_errors,
        diff_tokens=diff_tokens,
        speech_metrics=speech_metrics,
        phonetic_patterns=phonetic_patterns
    )


def format_analysis_for_export(analysis: AnalysisResult) -> Dict[str, Any]:
    """
    Format analysis result for JSON export.
    
    Args:
        analysis: AnalysisResult object
        
    Returns:
        Dictionary suitable for JSON serialization
    """
    result = {
        "metrics": {
            "wer": round(analysis.wer, 4),
            "cer": round(analysis.cer, 4),
        },
        "error_breakdown": {
            "substitutions": analysis.error_breakdown.substitutions,
            "insertions": analysis.error_breakdown.insertions,
            "deletions": analysis.error_breakdown.deletions,
            "correct": analysis.error_breakdown.correct,
            "total_words": analysis.error_breakdown.total_words,
            "total_errors": analysis.error_breakdown.total_errors,
        },
        "word_errors": [
            {
                "type": e.error_type.value,
                "reference": e.reference_word,
                "hypothesis": e.hypothesis_word,
                "position": e.position
            }
            for e in analysis.word_errors
        ]
    }
    
    if analysis.speech_metrics:
        result["speech_metrics"] = {
            "speech_rate_wpm": round(analysis.speech_metrics.speech_rate_wpm, 2),
            "total_words": analysis.speech_metrics.total_words,
            "audio_duration_seconds": round(analysis.speech_metrics.audio_duration_seconds, 2),
        }
        if analysis.speech_metrics.pause_count is not None:
            result["speech_metrics"]["pause_count"] = analysis.speech_metrics.pause_count
            result["speech_metrics"]["total_pause_duration"] = round(
                analysis.speech_metrics.total_pause_duration, 2
            )
            result["speech_metrics"]["average_pause_duration"] = round(
                analysis.speech_metrics.average_pause_duration, 3
            )
    
    if analysis.phonetic_patterns:
        result["phonetic_patterns"] = [
            {
                "name": p.pattern_name,
                "description": p.description,
                "frequency": p.frequency,
                "clinical_significance": p.clinical_significance,
                "affected_words": [
                    {"reference": ref, "hypothesis": hyp}
                    for ref, hyp in p.affected_words
                ]
            }
            for p in analysis.phonetic_patterns
        ]
    
    return result
