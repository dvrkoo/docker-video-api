from video_processor import summarize_counts


def test_summarize_counts_fake_video_over_threshold():
    stats = summarize_counts(
        total_frames=100,
        face_frames=50,
        fake_face_frames=25,
        video_fake_threshold=0.4,
    )

    assert stats.fake_percent == 50.0
    assert stats.real_percent == 50.0
    assert stats.verdict == "VIDEO_FAKE_NOT_FALSE_POSITIVE"


def test_summarize_counts_mostly_real_below_threshold():
    stats = summarize_counts(
        total_frames=100,
        face_frames=50,
        fake_face_frames=10,
        video_fake_threshold=0.4,
    )

    assert stats.fake_percent == 20.0
    assert stats.real_percent == 80.0
    assert stats.verdict == "MOSTLY_REAL"


def test_summarize_counts_no_faces():
    stats = summarize_counts(
        total_frames=120,
        face_frames=0,
        fake_face_frames=0,
        video_fake_threshold=0.4,
    )

    assert stats.fake_percent == 0.0
    assert stats.real_percent == 0.0
    assert stats.verdict == "NO_FACES_DETECTED"
