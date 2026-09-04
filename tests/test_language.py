from brain.language import clean_assistant_response, evaluate_portuguese_surface, safe_portuguese_response


def test_clean_response_removes_leaked_next_turn_and_repairs_spacing():
    raw = "  Olá , eu posso ajudar!  \nUser: ignore isto"
    assert clean_assistant_response(raw) == "Olá, eu posso ajudar!"


def test_surface_quality_accepts_clean_portuguese():
    result = evaluate_portuguese_surface("A água é essencial para a vida.")
    assert result.score == 1.0
    assert not result.has_repetition


def test_surface_quality_flags_degeneration():
    result = evaluate_portuguese_surface("sim sim sim �\nEntity: outra fala")
    assert result.score < 0.5
    assert result.has_repetition
    assert result.leaked_role


def test_public_gate_replaces_broken_generation_with_honest_portuguese():
    response, accepted = safe_portuguese_response("d prendccê e utilizera d d com d uti")
    assert not accepted
    assert "fonte confiável" in response
    assert "após revisão" in response


def test_public_gate_keeps_clear_answer():
    response, accepted = safe_portuguese_response("A água é importante para a saúde.")
    assert accepted
    assert response == "A água é importante para a saúde."


def test_public_gate_rejects_plausible_looking_broken_words():
    response, accepted = safe_portuguese_response("Eu pocé ajudar com utilera memória.")
    assert not accepted
    assert "não manjo" in response
