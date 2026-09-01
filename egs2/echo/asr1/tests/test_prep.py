
import pytest

from scripts.common.nlp_refs import extract_entities_from_tags, parse_nlp
from scripts.common.ref_index import mentions_to_entities
from scripts.data.prepare_ami import merge_speakers, parse_words_xml
from scripts.data.prepare_slue import parse_ner
from scripts.data.prepare_tedlium import parse_stm

e22_nlp = """token|speaker|ts|endTs|punctuation|prepunctuation|case|tags
Good|0|||||UC|[]
morning|0|||||LC|[]
Acme|1|||||UC|['3:ORG']
Corp|1|||.||UC|['3:ORG']
saw|1|||||LC|[]
Acme|1|||||UC|['9:ORG']
twelve|1|||||LC|['10:CARDINAL']
"""


def test_parse_nlp_e22_and_tag_entities(tmp_path):
    p = tmp_path / "d.nlp"
    p.write_text(e22_nlp)
    rows = parse_nlp(p)
    assert [r["token"] for r in rows] == ["Good", "morning", "Acme", "Corp", "saw", "Acme", "twelve"]
    assert rows[2]["tags"] == ["3:ORG"] and rows[0]["tags"] == []
    entities, stats = extract_entities_from_tags(rows, "d")
    assert stats["n_span_ids"] == 3
    by_cat = {(e["category"], e["canonical_surface"]): [o["span"] for o in e["occurrences"]] for e in entities}
    # "Acme Corp" and "Acme" fold to different surfaces, so they stay separate entities
    assert by_cat[("ORG", "Acme Corp")] == [[2, 3]]
    assert by_cat[("ORG", "Acme")] == [[5, 5]]
    assert by_cat[("CARDINAL", "twelve")] == [[6, 6]]


def test_e22_noncontiguous_tag_raises(tmp_path):
    p = tmp_path / "d.nlp"
    p.write_text(e22_nlp.replace("['10:CARDINAL']", "['3:ORG']"))
    with pytest.raises(ValueError, match="non-contiguous"):
        extract_entities_from_tags(parse_nlp(p), "d")


ami_xml = """<?xml version="1.0" encoding="ISO-8859-1"?>
<nite:root nite:id="M.A.words" xmlns:nite="http://nite.sourceforge.net/">
   <w nite:id="w0" starttime="0.5" endtime="0.9">Okay</w>
   <w nite:id="w1" starttime="0.9" endtime="0.9" punc="true">.</w>
   <w nite:id="w2" starttime="2.0" endtime="2.4">project</w>
   <w nite:id="w3" starttime="2.4" endtime="2.9">Kowalski</w>
   <w nite:id="w4">untimed</w>
</nite:root>
"""


def test_ami_words_parse_and_merge(tmp_path):
    p = tmp_path / "M.A.words.xml"
    p.write_text(ami_xml)
    words, despaced = parse_words_xml(p, "A")
    assert [w["token"] for w in words] == ["Okay", "project", "Kowalski"] and despaced == 0
    other = [{"token": "yes", "start": 1.0, "end": 1.2, "speaker": "B"}]
    merged = merge_speakers([words, other])
    assert [w["token"] for w in merged] == ["Okay", "yes", "project", "Kowalski"]


def test_stm_parse(tmp_path):
    p = tmp_path / "t.stm"
    p.write_text(
        ";; comment\n"
        "talk1 1 spk1 17.8 28.8 <o,f0,male> hello acme corp\n"
        "talk1 1 spk1 5.0 10.0 <o,f0,male> ignore_time_segment_in_scoring\n"
        "talk1 1 spk1 1.0 4.0 <o,f0,male> we begin\n"
    )
    segs = parse_stm(p)
    assert [s["tokens"] for s in segs] == [["we", "begin"], ["hello", "acme", "corp"]]
    assert segs[0]["start"] == 1.0 and segs[1]["speaker"] == "spk1"


def test_slue_parse_ner():
    assert parse_ner("None") == [] and parse_ner("[]") == []
    assert parse_ner("[['GPE', 0, 5], ['PERSON', 10, 3]]") == [("GPE", 0, 5), ("PERSON", 10, 13)]


def test_mentions_to_entities_spans_and_grouping():
    tokens = ["we", "met", "Kai", "Fu", "Lee", "and", "later", "kai", "fu", "lee", "again"]
    text = " ".join(tokens)
    a, b = text.index("Kai"), text.index("Kai") + len("Kai Fu Lee")
    c, d = text.index("kai"), text.index("kai") + len("kai fu lee")
    meta = [{"start": float(i), "end": i + 0.5, "speaker": "s1"} for i in range(len(tokens))]
    entities = mentions_to_entities(tokens, [("PERSON", a, b), ("PERSON", c, d)], meta)
    # both mentions fold to one entity with two occurrences at the right token spans
    assert len(entities) == 1
    e = entities[0]
    assert e["category"] == "PERSON" and e["canonical_surface"] in ("Kai Fu Lee", "kai fu lee")
    assert [o["span"] for o in e["occurrences"]] == [[2, 4], [7, 9]]
    assert e["occurrences"][0]["start"] == 2.0 and e["occurrences"][0]["end"] == 4.5
    # a mention that starts mid-token still snaps to the covering token
    assert mentions_to_entities(tokens, [("ORG", a + 1, b)])[0]["occurrences"][0]["span"] == [2, 4]
