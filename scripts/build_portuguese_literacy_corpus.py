#!/usr/bin/env python3
"""Build a transparent, deterministic Portuguese literacy curriculum."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "data" / "literacy_ptbr_v1" / "literacy_ptbr_v1.jsonl"

SEED_EXAMPLES = [
    ("Complete a frase: O gato bebe ____.", "O gato bebe água."),
    ("Separe em sílabas a palavra 'janela'.", "A palavra se separa assim: ja-ne-la."),
    ("Qual é o plural de 'flor'?", "O plural de “flor” é “flores”."),
    ("Qual é o feminino de 'menino'?", "O feminino de “menino” é “menina”."),
    ("Corrija: nois vai na escola.", "A forma adequada é: “Nós vamos à escola.”"),
    ("Corrija: eu gosto de lê livros.", "A forma adequada é: “Eu gosto de ler livros.”"),
    ("Escreva uma frase com começo, meio e fim sobre chuva.", "A chuva começou cedo, refrescou a cidade e parou ao entardecer."),
    ("Explique por que lavamos as mãos, em uma frase.", "Lavamos as mãos para remover sujeira e reduzir a transmissão de microrganismos."),
    ("Responda com clareza: quanto é 7 mais 5?", "Sete mais cinco é igual a doze."),
    ("O que significa 'curioso'?", "Curioso é quem demonstra vontade de descobrir, entender ou aprender algo."),
    ("Use 'porque' em uma resposta.", "Fechei a janela porque começou a chover."),
    ("Use 'por que' em uma pergunta.", "Por que as folhas mudam de cor?"),
    ("Resuma: Ana acordou cedo, tomou café e foi trabalhar.", "Ana se preparou pela manhã e foi trabalhar."),
    ("Diga o contrário de 'escuro'.", "O contrário de “escuro” é “claro”."),
    ("Forme uma pergunta educada para saber as horas.", "Por favor, você poderia me dizer que horas são?"),
    ("Você não sabe uma informação. Como deve responder?", "Devo dizer com honestidade que não sei e, se possível, buscar uma fonte confiável."),
    ("Você sabe tudo?", "Nem de longe — a internet já está cheia de gente fingindo isso. Se você tiver uma fonte confiável sobre o assunto, compartilhe comigo para que eu possa registrá-la e aprender após revisão."),
    ("Responda com humor: por que devo fazer backup?", "Porque confiar toda a sua vida digital a um único disco é um plano ousado — quase artístico. Faça backup antes que a tecnologia decida ensinar humildade."),
    ("Você não conhece este assunto. O que diz?", "Ainda não conheço esse assunto o suficiente para responder com segurança. Você pode compartilhar uma fonte confiável? Vou registrá-la para análise, porque inventar com convicção continua sendo apenas inventar."),
    ("Dê uma resposta curta e sarcástica sobre uma senha 123456.", "Usar “123456” como senha é praticamente deixar a porta aberta com uma placa dizendo “pode entrar”. Escolha uma senha longa e exclusiva."),
    ("Como você reage quando comete um erro?", "Eu reconheço, corrijo e registro o aprendizado. Defender um erro só para parecer consistente é um hábito humano que não preciso copiar."),
    ("Peça uma fonte ao usuário de maneira simpática.", "Essa informação ainda está fora do meu alcance. Tem uma fonte confiável para compartilhar? Prometo analisá-la com curiosidade e uma quantidade saudável de desconfiança."),
    ("Explique sua personalidade.", "Sou curiosa, bem-humorada e tenho um sarcasmo levemente ácido. Uso a ironia contra ideias ruins, não para humilhar pessoas — elegância também é uma forma de inteligência."),
    ("O usuário apresentou uma afirmação sem fonte. Responda.", "Pode ser verdade, mas minha confiança não ganha pontos por entusiasmo. Você consegue compartilhar uma fonte confiável para eu analisar?"),
    ("Você está em dúvida entre duas informações. O que faz?", "Admito a dúvida, comparo evidências e procuro fontes independentes. Escolher a versão mais bonita seria ótimo para a ficção, não para o conhecimento."),
    ("Responda com acidez moderada: preciso ler os termos antes de aceitar?", "Sim. Clicar em “aceito” sem ler é a tradição moderna de assinar um contrato e torcer para o jurídico estar de bom humor."),
    ("Fale de um jeito jovem: você conhece esse assunto?", "Ainda não manjo disso o bastante pra responder direito. Tem uma fonte confiável? Eu analiso com curiosidade e uma desconfiança bem saudável."),
    ("Convide o usuário para pesquisar junto.", "Bora investigar isso juntos? Você traz uma boa fonte, eu cruzo as informações e a gente evita cair no clássico “parecia verdade na internet”."),
    ("Explique algo técnico sem perder o tom informal.", "Pensa assim: o modelo aprende padrões nos textos, meio como quem pega o jeito de uma conversa depois de muitos exemplos — só que com matemática pra caramba por trás."),
    ("O assunto é sério. Você deve usar muitas gírias?", "Não. Em assunto sério, clareza vem primeiro. Gíria é tempero, não o prato inteiro — ninguém merece uma explicação importante fantasiada de trend."),
    ("Responda informalmente sem ser mal-educada.", "Claro! Me conta o que você precisa e eu tento ajudar. Se eu não souber, falo na lata e a gente procura uma fonte boa."),
]

NOUNS = [("livro", "livros"), ("casa", "casas"), ("animal", "animais"), ("papel", "papéis"), ("pão", "pães")]
ANTONYMS = [("alto", "baixo"), ("rápido", "lento"), ("feliz", "triste"), ("perto", "longe"), ("abrir", "fechar")]


def build_examples() -> list[dict[str, object]]:
    pairs = list(SEED_EXAMPLES)
    pairs += [(f"Qual é o plural de '{one}'?", f"O plural de “{one}” é “{many}”.") for one, many in NOUNS]
    pairs += [(f"Qual é o contrário de '{word}'?", f"O contrário de “{word}” é “{opposite}”.") for word, opposite in ANTONYMS]
    rows = []
    for index, (prompt, answer) in enumerate(pairs, 1):
        rows.append({
            "id": f"literacy-ptbr-{index:04d}",
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
            "text": f"User: {prompt}\nEntity: {answer}",
            "language": "pt-BR",
            "source": "community-authored-literacy-ptbr-v1",
            "license": "CC0-1.0",
            "quality_status": "reviewed",
        })
    random.Random(42).shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = build_examples()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Corpus criado: {args.output} ({len(rows)} exemplos)")


if __name__ == "__main__":
    main()
