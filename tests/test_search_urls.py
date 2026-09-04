from research.web_search import normalize_search_result_url


def test_unwraps_duckduckgo_result_url():
    redirect = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fnexusdc.com.br%2F%3Fa%3D1&rut=abc"
    assert normalize_search_result_url(redirect) == "https://nexusdc.com.br/?a=1"


def test_keeps_direct_result_url():
    assert normalize_search_result_url("https://pt.wikipedia.org/wiki/L") == "https://pt.wikipedia.org/wiki/L"
