"""
Tests for Error Analysis and Metrics Module

Tests cover WER/CER calculation, error breakdown, speech metrics,
and phonetic pattern detection.
"""

import pytest

from src.analysis_utils import (
    AnalysisResult,
    DiffToken,
    ErrorBreakdown,
    ErrorType,
    PhoneticPattern,
    SpeechMetrics,
    WordError,
    analyze_transcription,
    compute_cer,
    compute_speech_rate,
    compute_speech_rate_with_pauses,
    compute_wer,
    format_analysis_for_export,
    generate_diff_tokens,
    get_error_breakdown,
    get_word_errors,
    identify_phonetic_patterns,
    normalize_text,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def perfect_match():
    """Ground truth and transcription that match perfectly."""
    return "the quick brown fox", "the quick brown fox"


@pytest.fixture
def one_substitution():
    """One word substituted."""
    return "the quick brown fox", "the quick brown box"


@pytest.fixture
def one_insertion():
    """One extra word in transcription."""
    return "the quick fox", "the quick brown fox"


@pytest.fixture
def one_deletion():
    """One word missing from transcription."""
    return "the quick brown fox", "the quick fox"


@pytest.fixture
def complex_errors():
    """Multiple types of errors."""
    return (
        "the quick brown fox jumps over the lazy dog",
        "a quick brown box jump over lazy dogs"
    )


# ============================================================================
# Tests for normalize_text
# ============================================================================

class TestNormalizeText:
    def test_lowercase(self):
        """Test text is converted to lowercase."""
        assert normalize_text("Hello World") == "hello world"
    
    def test_remove_punctuation(self):
        """Test punctuation is removed."""
        assert normalize_text("Hello, World!") == "hello world"
    
    def test_preserve_apostrophes(self):
        """Test apostrophes are preserved for contractions."""
        assert normalize_text("don't won't") == "don't won't"
    
    def test_normalize_whitespace(self):
        """Test multiple spaces are normalized."""
        assert normalize_text("hello    world") == "hello world"
    
    def test_empty_string(self):
        """Test empty string handling."""
        assert normalize_text("") == ""


# ============================================================================
# Tests for compute_wer
# ============================================================================

class TestComputeWER:
    def test_perfect_match(self, perfect_match):
        """Test WER is 0 for perfect match."""
        gt, trans = perfect_match
        assert compute_wer(gt, trans) == 0.0
    
    def test_one_substitution(self, one_substitution):
        """Test WER for one substitution."""
        gt, trans = one_substitution
        wer_score = compute_wer(gt, trans)
        # 1 error out of 4 words = 0.25
        assert wer_score == pytest.approx(0.25, abs=0.01)
    
    def test_one_insertion(self, one_insertion):
        """Test WER for one insertion."""
        gt, trans = one_insertion
        wer_score = compute_wer(gt, trans)
        # 1 error out of 3 words = 0.333
        assert wer_score == pytest.approx(0.333, abs=0.01)
    
    def test_one_deletion(self, one_deletion):
        """Test WER for one deletion."""
        gt, trans = one_deletion
        wer_score = compute_wer(gt, trans)
        # 1 error out of 4 words = 0.25
        assert wer_score == pytest.approx(0.25, abs=0.01)
    
    def test_complete_mismatch(self):
        """Test WER for completely different texts."""
        wer_score = compute_wer("hello world", "foo bar")
        # 2 substitutions out of 2 words = 1.0
        assert wer_score == pytest.approx(1.0, abs=0.01)
    
    def test_empty_ground_truth(self):
        """Test WER with empty ground truth."""
        assert compute_wer("", "hello") == 1.0
        assert compute_wer("", "") == 0.0
    
    def test_case_insensitive(self):
        """Test WER is case insensitive."""
        assert compute_wer("Hello World", "hello world") == 0.0


# ============================================================================
# Tests for compute_cer
# ============================================================================

class TestComputeCER:
    def test_perfect_match(self, perfect_match):
        """Test CER is 0 for perfect match."""
        gt, trans = perfect_match
        assert compute_cer(gt, trans) == 0.0
    
    def test_one_character_diff(self):
        """Test CER for one character difference."""
        cer_score = compute_cer("cat", "bat")
        # 1 character error out of 3 = 0.333
        assert cer_score == pytest.approx(0.333, abs=0.01)
    
    def test_empty_ground_truth(self):
        """Test CER with empty ground truth."""
        assert compute_cer("", "hello") == 1.0


# ============================================================================
# Tests for get_error_breakdown
# ============================================================================

class TestGetErrorBreakdown:
    def test_perfect_match(self, perfect_match):
        """Test breakdown for perfect match."""
        gt, trans = perfect_match
        breakdown = get_error_breakdown(gt, trans)
        
        assert breakdown.substitutions == 0
        assert breakdown.insertions == 0
        assert breakdown.deletions == 0
        assert breakdown.correct == 4
        assert breakdown.total_words == 4
    
    def test_one_substitution(self, one_substitution):
        """Test breakdown with one substitution."""
        gt, trans = one_substitution
        breakdown = get_error_breakdown(gt, trans)
        
        assert breakdown.substitutions == 1
        assert breakdown.insertions == 0
        assert breakdown.deletions == 0
        assert breakdown.correct == 3
    
    def test_one_insertion(self, one_insertion):
        """Test breakdown with one insertion."""
        gt, trans = one_insertion
        breakdown = get_error_breakdown(gt, trans)
        
        assert breakdown.insertions == 1
        assert breakdown.substitutions == 0
        assert breakdown.deletions == 0
    
    def test_one_deletion(self, one_deletion):
        """Test breakdown with one deletion."""
        gt, trans = one_deletion
        breakdown = get_error_breakdown(gt, trans)
        
        assert breakdown.deletions == 1
        assert breakdown.substitutions == 0
        assert breakdown.insertions == 0
    
    def test_total_errors(self, complex_errors):
        """Test total_errors property."""
        gt, trans = complex_errors
        breakdown = get_error_breakdown(gt, trans)
        
        expected_total = breakdown.substitutions + breakdown.insertions + breakdown.deletions
        assert breakdown.total_errors == expected_total
    
    def test_error_rates(self, one_substitution):
        """Test error rate calculations."""
        gt, trans = one_substitution
        breakdown = get_error_breakdown(gt, trans)
        
        assert breakdown.substitution_rate == pytest.approx(0.25, abs=0.01)
        assert breakdown.insertion_rate == 0.0
        assert breakdown.deletion_rate == 0.0


# ============================================================================
# Tests for get_word_errors
# ============================================================================

class TestGetWordErrors:
    def test_perfect_match(self, perfect_match):
        """Test word errors for perfect match (all MATCH type)."""
        gt, trans = perfect_match
        errors = get_word_errors(gt, trans)
        
        assert len(errors) == 4
        assert all(e.error_type == ErrorType.MATCH for e in errors)
    
    def test_substitution_detected(self, one_substitution):
        """Test substitution is detected."""
        gt, trans = one_substitution
        errors = get_word_errors(gt, trans)
        
        substitutions = [e for e in errors if e.error_type == ErrorType.SUBSTITUTION]
        assert len(substitutions) == 1
        assert substitutions[0].reference_word == "fox"
        assert substitutions[0].hypothesis_word == "box"
    
    def test_insertion_detected(self, one_insertion):
        """Test insertion is detected."""
        gt, trans = one_insertion
        errors = get_word_errors(gt, trans)
        
        insertions = [e for e in errors if e.error_type == ErrorType.INSERTION]
        assert len(insertions) == 1
        assert insertions[0].hypothesis_word == "brown"
    
    def test_deletion_detected(self, one_deletion):
        """Test deletion is detected."""
        gt, trans = one_deletion
        errors = get_word_errors(gt, trans)
        
        deletions = [e for e in errors if e.error_type == ErrorType.DELETION]
        assert len(deletions) == 1
        assert deletions[0].reference_word == "brown"
    
    def test_word_error_structure(self, one_substitution):
        """Test WordError has all required fields."""
        gt, trans = one_substitution
        errors = get_word_errors(gt, trans)
        
        for error in errors:
            assert hasattr(error, 'error_type')
            assert hasattr(error, 'reference_word')
            assert hasattr(error, 'hypothesis_word')
            assert hasattr(error, 'position')


# ============================================================================
# Tests for generate_diff_tokens
# ============================================================================

class TestGenerateDiffTokens:
    def test_perfect_match(self, perfect_match):
        """Test diff tokens for perfect match."""
        gt, trans = perfect_match
        tokens = generate_diff_tokens(gt, trans)
        
        assert len(tokens) == 4
        assert all(t.error_type == ErrorType.MATCH for t in tokens)
    
    def test_substitution_token(self, one_substitution):
        """Test diff token for substitution includes reference."""
        gt, trans = one_substitution
        tokens = generate_diff_tokens(gt, trans)
        
        sub_tokens = [t for t in tokens if t.error_type == ErrorType.SUBSTITUTION]
        assert len(sub_tokens) == 1
        assert sub_tokens[0].text == "box"
        assert sub_tokens[0].reference == "fox"
    
    def test_insertion_token(self, one_insertion):
        """Test diff token for insertion."""
        gt, trans = one_insertion
        tokens = generate_diff_tokens(gt, trans)
        
        ins_tokens = [t for t in tokens if t.error_type == ErrorType.INSERTION]
        assert len(ins_tokens) == 1
        assert ins_tokens[0].text == "brown"
    
    def test_deletion_token(self, one_deletion):
        """Test diff token for deletion."""
        gt, trans = one_deletion
        tokens = generate_diff_tokens(gt, trans)
        
        del_tokens = [t for t in tokens if t.error_type == ErrorType.DELETION]
        assert len(del_tokens) == 1
        assert del_tokens[0].text == "brown"


# ============================================================================
# Tests for speech metrics
# ============================================================================

class TestSpeechMetrics:
    def test_compute_speech_rate(self):
        """Test speech rate calculation."""
        transcription = "one two three four five"  # 5 words
        duration = 30.0  # 30 seconds
        
        metrics = compute_speech_rate(transcription, duration)
        
        assert metrics.speech_rate_wpm == pytest.approx(10.0, abs=0.1)
        assert metrics.total_words == 5
        assert metrics.audio_duration_seconds == 30.0
    
    def test_speech_rate_with_pauses(self):
        """Test speech rate with pause information."""
        transcription = "one two three four five"
        duration = 30.0
        
        metrics = compute_speech_rate_with_pauses(
            transcription,
            duration,
            pause_count=3,
            total_pause_duration=5.0
        )
        
        assert metrics.pause_count == 3
        assert metrics.total_pause_duration == 5.0
        assert metrics.average_pause_duration == pytest.approx(5.0/3, abs=0.01)
    
    def test_zero_duration(self):
        """Test handling of zero duration."""
        metrics = compute_speech_rate("hello world", 0.0)
        assert metrics.speech_rate_wpm == 0.0
    
    def test_empty_transcription(self):
        """Test handling of empty transcription."""
        metrics = compute_speech_rate("", 30.0)
        assert metrics.total_words == 0
        assert metrics.speech_rate_wpm == 0.0


# ============================================================================
# Tests for phonetic pattern analysis
# ============================================================================

class TestPhoneticPatterns:
    def test_no_patterns_for_perfect_match(self, perfect_match):
        """Test no patterns detected for perfect match."""
        gt, trans = perfect_match
        errors = get_word_errors(gt, trans)
        patterns = identify_phonetic_patterns(errors)
        
        assert len(patterns) == 0
    
    def test_pattern_structure(self):
        """Test PhoneticPattern has all required fields."""
        # Create a scenario with fricative errors
        errors = [
            WordError(
                error_type=ErrorType.SUBSTITUTION,
                reference_word="fish",
                hypothesis_word="fih",
                position=0
            )
        ]
        patterns = identify_phonetic_patterns(errors)
        
        for pattern in patterns:
            assert hasattr(pattern, 'pattern_name')
            assert hasattr(pattern, 'description')
            assert hasattr(pattern, 'affected_words')
            assert hasattr(pattern, 'frequency')
            assert hasattr(pattern, 'clinical_significance')


# ============================================================================
# Tests for complete analysis
# ============================================================================

class TestAnalyzeTranscription:
    def test_basic_analysis(self, one_substitution):
        """Test basic analysis without audio duration."""
        gt, trans = one_substitution
        result = analyze_transcription(gt, trans)
        
        assert isinstance(result, AnalysisResult)
        assert result.wer == pytest.approx(0.25, abs=0.01)
        assert result.error_breakdown.substitutions == 1
        assert len(result.word_errors) == 4
        assert len(result.diff_tokens) == 4
    
    def test_analysis_with_duration(self, perfect_match):
        """Test analysis with audio duration."""
        gt, trans = perfect_match
        result = analyze_transcription(gt, trans, audio_duration_seconds=10.0)
        
        assert result.speech_metrics is not None
        assert result.speech_metrics.speech_rate_wpm > 0
    
    def test_analysis_with_pause_info(self, perfect_match):
        """Test analysis with pause information."""
        gt, trans = perfect_match
        pause_info = {
            'num_pauses': 2,
            'total_pause_duration': 1.5
        }
        result = analyze_transcription(
            gt, trans,
            audio_duration_seconds=10.0,
            pause_info=pause_info
        )
        
        assert result.speech_metrics.pause_count == 2
        assert result.speech_metrics.total_pause_duration == 1.5
    
    def test_analysis_result_fields(self, complex_errors):
        """Test AnalysisResult has all required fields."""
        gt, trans = complex_errors
        result = analyze_transcription(gt, trans, audio_duration_seconds=30.0)
        
        assert hasattr(result, 'wer')
        assert hasattr(result, 'cer')
        assert hasattr(result, 'error_breakdown')
        assert hasattr(result, 'word_errors')
        assert hasattr(result, 'diff_tokens')
        assert hasattr(result, 'speech_metrics')
        assert hasattr(result, 'phonetic_patterns')


# ============================================================================
# Tests for export formatting
# ============================================================================

class TestFormatAnalysisForExport:
    def test_export_structure(self, complex_errors):
        """Test exported format has expected structure."""
        gt, trans = complex_errors
        result = analyze_transcription(gt, trans, audio_duration_seconds=30.0)
        exported = format_analysis_for_export(result)
        
        assert 'metrics' in exported
        assert 'error_breakdown' in exported
        assert 'word_errors' in exported
        assert 'speech_metrics' in exported
    
    def test_metrics_rounded(self, one_substitution):
        """Test metrics are properly rounded."""
        gt, trans = one_substitution
        result = analyze_transcription(gt, trans)
        exported = format_analysis_for_export(result)
        
        # WER should be rounded to 4 decimal places
        assert exported['metrics']['wer'] == round(result.wer, 4)
    
    def test_word_errors_serializable(self, complex_errors):
        """Test word errors are JSON serializable."""
        gt, trans = complex_errors
        result = analyze_transcription(gt, trans)
        exported = format_analysis_for_export(result)
        
        for error in exported['word_errors']:
            assert 'type' in error
            assert 'reference' in error
            assert 'hypothesis' in error
            assert 'position' in error
            # Type should be string, not Enum
            assert isinstance(error['type'], str)
    
    def test_export_without_speech_metrics(self, one_substitution):
        """Test export works without speech metrics."""
        gt, trans = one_substitution
        result = analyze_transcription(gt, trans)  # No duration
        exported = format_analysis_for_export(result)
        
        # Should not have speech_metrics key
        assert 'speech_metrics' not in exported
