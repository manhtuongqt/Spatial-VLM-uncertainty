"""Tests for phrase identity, position, fail-closed scope and legacy isolation."""
from pcrau.query_parser import parse_query
from pcrau.text import tokenize, words


def test_phrase_spans_distinguish_repeated_words():
    prompt = "Apple is mentioned here. Please locate the apple that is right of the purple cube. Output (x,y)."
    q = parse_query(prompt)
    assert q.supported and q.predicate == "right_of" and q.reference_frame == "image"
    assert prompt[q.target.char_start:q.target.char_end] == "apple"
    assert q.target.token_start == 7
    anchor = q.anchors[0]
    assert prompt[anchor.char_start:anchor.char_end] == "purple cube"
    assert words(prompt)[anchor.token_start:anchor.token_end] == ["purple", "cube"]
    assert not q.token_masks()["target_token_mask"][0]


def test_swap_changes_noun_roles():
    a = parse_query("locate the apple that is right of the purple cube.")
    b = parse_query("locate the purple cube that is left of the apple.")
    assert a.target.text == b.anchors[0].text == "apple"
    assert b.target.text == a.anchors[0].text == "purple cube"
    assert a.predicate != b.predicate


def test_prefixes_shift_spans_without_changing_identity():
    prompt = "locate the apple that is right of the purple cube."
    original = parse_query(prompt)
    for prefix in ("", "Please ", "In this scene, ", "Can you "):
        q = parse_query(prefix + prompt)
        assert q.target.text == original.target.text and q.anchors[0].text == original.anchors[0].text
        assert q.target.token_start == original.target.token_start + len(words(prefix))
        ids, real = tokenize(prefix+prompt,72,8192)
        mask = q.token_masks()["anchor_token_masks"][0]
        phrase_ids,_ = tokenize("purple cube",2,8192)
        assert [v for v,m in zip(ids,mask) if m] == phrase_ids
        assert all(not m or r for m,r in zip(mask,real))


def test_direct_has_target_and_no_anchor():
    q = parse_query("Please locate the fruit. Output the target pixel.")
    assert q.supported and q.predicate == "direct" and q.reference_frame is None
    assert q.target.text == "fruit" and q.anchors == ()
    assert not any(q.token_masks()["anchor_slot_mask"])
    assert not any(any(m) for m in q.token_masks()["anchor_token_masks"])


def test_unsupported_queries_never_produce_conditioning_masks():
    prompts = [
        "locate the apple that is behind the cube.",
        "locate the apple above the cube.",
        "locate the apple next to the cube.",
        "locate the apple near the cube.",
        "locate the apple between the cube and the lemon in camera depth.",
        "locate the apple that is right of the cube and left of the lemon.",
        "locate the apple that is not right of the cube.",
        "locate the apple that is right of the cube. It must be left of the lemon.",
        "locate the apple that is right of the cube in the world frame.",
        "Do not locate the apple.", "locate the apple or the lemon.",
        "locate the apple that is right of the.", "", "Find the apple.",
        "locate the café.", "locate the İtem.",
    ]
    for prompt in prompts:
        q = parse_query(prompt)
        assert not q.supported, prompt
        assert q.predicate is None and q.target is None and q.anchors == ()
        masks = q.token_masks()
        assert not any(masks["target_token_mask"])
        assert not any(any(m) for m in masks["anchor_token_masks"])


def test_multiple_commands_are_ambiguous():
    q = parse_query("locate the apple. locate the cube.")
    assert q.status == "ambiguous" and q.reason == "multiple_locate_commands"


def test_required_phrase_truncation_is_rejected():
    prompt = "locate the apple that is right of the purple cube."
    q = parse_query(prompt,max_tokens=9)
    assert not q.supported and q.reason == "required_phrase_truncated"
    complete = parse_query(prompt + " " + "explanation " * 100,max_tokens=72)
    assert complete.supported and complete.total_tokens > 72


def test_case_spacing_and_hyphen_align_to_legacy_words():
    prompt = "PLEASE locate the  RED-CUBE that is LEFT of the green cube!"
    q = parse_query(prompt)
    assert q.supported and q.target.text == "red cube"
    assert prompt[q.target.char_start:q.target.char_end] == "RED-CUBE"
    assert words(prompt)[q.target.token_start:q.target.token_end] == ["red","cube"]


def test_anchor_capacity_and_invalid_arguments():
    q = parse_query("locate the apple that is right of the cube.",max_anchors=0)
    assert not q.supported and q.reason == "anchor_capacity"
    assert parse_query("locate the apple.",max_anchors=0).supported
    for args in ({"max_tokens":0},{"max_anchors":-1}):
        try:
            parse_query("locate the apple.",**args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid dimensions accepted")
