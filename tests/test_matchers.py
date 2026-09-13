from evals.matchers import match_args, match_calls, match_value, squash, support_ratio, content_stems


def test_squash_ignores_case_punctuation_and_superscripts():
    assert squash("SYN-ACK") == squash("syn ack") == "synack"
    assert squash("wL²/8") == squash("wl^2 / 8")


def test_contains_accepts_list_of_terms():
    assert match_value({"$contains": ["1200", "35"]}, "0.5*1200*3.5**2")
    assert not match_value({"$contains": ["1200", "99"]}, "0.5*1200*3.5**2")


def test_one_of_and_range():
    assert match_value({"$one_of": ["mcq", "short"]}, "short")
    assert match_value({"$range": [1, 5]}, 3)
    assert not match_value({"$range": [1, 5]}, "3")


def test_numbers_do_not_match_booleans():
    assert match_value(1, 1.0)
    assert not match_value(1, True)


def test_items_contain_and_min_items():
    rows = [["Doric", "plain"], ["Ionic", "volute"], ["Corinthian", "acanthus"]]
    assert match_value({"$all": [{"$min_items": 3}, {"$items_contain": ["doric", "corinthian"]}]}, rows)
    assert not match_value({"$min_items": 4}, rows)


def test_optional_key_may_be_absent_but_must_match_when_present():
    expected = {"top_n": 5, "subject": {"$optional": {"$contains": "anatomy"}}}
    assert match_args(expected, {"top_n": 5})
    assert match_args(expected, {"top_n": 5, "subject": "Human Anatomy"})
    assert not match_args(expected, {"top_n": 5, "subject": "physics"})
    assert not match_args(expected, None)


def test_match_calls_is_order_insensitive_and_separates_selection_from_args():
    expected = [{"name": "export_notes", "args": {"format": "markdown"}}, {"name": "export_notes", "args": {"format": "pdf"}}]
    assert match_calls(expected, [("export_notes", {"format": "pdf"}), ("export_notes", {"format": "markdown"})]) == (True, True)
    assert match_calls(expected, [("export_notes", {"format": "pdf"}), ("export_notes", {"format": "pdf"})]) == (True, False)
    assert match_calls(expected, [("export_notes", {"format": "pdf"})]) == (False, False)


def test_grounded_scoring_separates_refusal_from_mixed_replies():
    from evals.suites.grounded import _score

    answerable = {"answerable": True, "answers": ["wL²/8"]}
    unanswerable = {"answerable": False, "answers": []}
    assert _score(answerable, "It is wL^2/8.") == {"refused": False, "mixed": False, "correct": True}
    assert _score(answerable, "It is wL²/8. NOT_IN_SOURCE") == {"refused": False, "mixed": True, "correct": False}
    assert _score(unanswerable, "NOT_IN_SOURCE") == {"refused": True, "mixed": False, "correct": True}
    assert _score(unanswerable, "Probably 1841.")["correct"] is False


def test_cer_ignores_case_and_punctuation_and_is_capped():
    from evals.matchers import cer

    assert cer("Kirchhoff's law: V = IR.", "kirchhoff s law v ir") == 0.0
    assert cer("abcd", "abed") == 0.25
    assert cer("abc", "a completely different and much longer string") == 1.0
    assert cer("", "") == 0.0


def test_support_ratio_counts_passage_words():
    source = content_stems("Flying buttresses carry outward thrust to buttress piers.")
    assert support_ratio("buttresses carry thrust", source) == 1.0
    assert support_ratio("the", source) is None
