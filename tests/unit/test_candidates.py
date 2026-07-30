from container_id.iso6346.candidates import parse_candidate, score_candidate


def test_parse_candidate_valid() -> None:
    cid = parse_candidate("CSQU3054383")
    assert cid is not None
    assert cid.owner_code == "CSQ"
    assert cid.category_id == "U"
    assert cid.serial_number == "305438"
    assert cid.check_digit == 3

def test_parse_candidate_without_check_digit() -> None:
    cid = parse_candidate("CSQU305438")
    assert cid is not None
    assert cid.check_digit is None

def test_parse_candidate_invalid_length() -> None:
    assert parse_candidate("CSQU30543") is None
    assert parse_candidate("CSQU30543834") is None

def test_parse_candidate_invalid_structure() -> None:
    assert parse_candidate("123U3054383") is None
    assert parse_candidate("CSQ13054383") is None
    assert parse_candidate("CSQUABCDEF3") is None

def test_score_candidate() -> None:
    valid_cid = parse_candidate("CSQU3054383")
    assert valid_cid is not None
    score = score_candidate(valid_cid)
    assert score > 0

    invalid_check_cid = parse_candidate("CSQU3054384")
    assert invalid_check_cid is not None
    assert score_candidate(invalid_check_cid, require_valid_check_digit=True) == -1

    # Check that it returns a penalty when not required
    score_invalid_lenient = score_candidate(invalid_check_cid, require_valid_check_digit=False)
    assert score_invalid_lenient < score
