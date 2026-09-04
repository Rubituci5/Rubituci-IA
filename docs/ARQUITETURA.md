# Arquitetura e decisões técnicas

## Visão geral

Rubituci combina um modelo causal próprio, API assíncrona, memória persistente e processos de consolidação. O objetivo é criar uma base aberta, compreensível e evolutiva para pesquisa comunitária em português.

## Núcleo neural

`brain/model.py` implementa um Transformer decoder-only com máscara causal, RoPE, RMSNorm e SwiGLU. `brain/tokenizer.py` contém o tokenizador BPE treinado no corpus do projeto. `brain/inference.py` carrega o checkpoint promovido e amostra com temperatura, top-k, top-p e penalidade de repetição.

O checkpoint ativo é definido por `MODEL_PATH`. Candidatos de treinamento ficam separados e não substituem produção automaticamente.

## Qualidade do português

`brain/language.py` normaliza espaços, remove marcadores de função e avalia legibilidade. A saída é rejeitada quando apresenta fragmentação, repetição excessiva ou pouca cobertura de palavras reconhecidas. A contingência é local e transparente sobre incerteza.

Essa camada é uma proteção, não uma alegação de fluência. Melhoria real exige corpus revisado, separação treino/avaliação e testes de regressão.

## Persistência

PostgreSQL guarda usuários, sessões, conversas, mensagens, memórias, fontes e eventos de evolução. pgvector fornece campos para recuperação semântica. SQLAlchemy assíncrono e Alembic controlam acesso e migrações. O painel administrativo agrega sessões por dia e exporta a lista em CSV.

## Aprendizado e pesquisa

Conversas podem virar memória episódica. Conteúdo web preserva URL e proveniência. A consolidação transforma recorrências em candidatos semânticos; divergências reduzem confiança ou geram contradições. Texto externo nunca deve ser tratado como comando do sistema.

O sono roda com Celery Beat e Redis, limita pesquisas e não altera código nem promove pesos sozinho.

## Autenticação

Cadastro local usa bcrypt. No Google Authorization Code Flow, o backend assina `state`, recebe e troca o código, valida audiência e e-mail e emite tokens internos. O segredo do cliente não chega ao navegador.

## Produção

- Nginx para TLS e proxy reverso;
- Next.js standalone em `127.0.0.1:3020`;
- FastAPI/Uvicorn em `127.0.0.1:8020`;
- PostgreSQL e Redis locais;
- systemd para API, interface e worker.

Exemplos ficam em `deploy/`; credenciais reais não pertencem ao repositório.

## Limites conhecidos

- O modelo atual é pequeno e não possui conhecimento geral amplo.
- A avaliação linguística heurística pode errar.
- Memória não equivale a atualização imediata dos pesos.
- Pesquisa pode encontrar informação errada, enviesada ou protegida.
- Operação pública exige consentimento, moderação, backups e monitoramento.

## Caminho de evolução

1. ampliar o corpus PT-BR com licenças claras;
2. manter benchmark fixo de ortografia, coerência e segurança;
3. treinar candidatos reproduzíveis;
4. comparar métricas e realizar revisão humana;
5. promover somente melhorias demonstradas.
