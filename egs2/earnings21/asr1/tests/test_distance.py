from scripts.common.distance import combined_distance, d_lev_norm, d_phon


def test_lev_norm():
    assert d_lev_norm("kowalski", "kowalski") == 0.0
    assert d_lev_norm("kowalski", "kowalsky") == 1 / 8
    assert d_lev_norm("", "abc") == 1.0
    assert d_lev_norm("", "") == 0.0


def test_phon_ordering():
    assert d_phon("kowalski", "kowalsky") < 0.15
    # panphon feature costs are fractions of the feature vector, so absolute values
    # sit lower than plain phone edit distance; 0.25 is far on this scale
    assert d_phon("kowalski", "barclays") > 0.25
    assert d_phon("kowalski", "kowalski") == 0.0


def test_phon_feature_weighting():
    # /b/~/p/ differs in one feature bundle, /b/~/ʃ/ in several
    assert d_phon("bat", "pat") < d_phon("bat", "shat")


def test_phon_empty():
    assert d_phon("...", "...") == 0.0
    assert d_phon("...", "kowalski") == 1.0


def test_combined():
    close = combined_distance("kowalski", "kowalsky")
    far = combined_distance("kowalski", "barclays")
    assert close < 0.2 < far
    assert combined_distance("kowalski", "kowalski") == 0.0
    # the tiny_doc fixture trap: two genuinely different entities under the default
    # merge threshold of 0.35, which is what cluster purity has to catch
    assert combined_distance("meridian", "veridian") < 0.35
