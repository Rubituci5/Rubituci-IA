# Rubituci IA

Rubituci é uma inteligência artificial experimental, aberta e independente, criada para conversar em português brasileiro, aprender com conteúdos revisados e construir uma memória própria ao longo do tempo.

O projeto não usa um grande modelo comercial escondido por trás da interface. O núcleo atual é um Transformer causal pequeno, implementado e treinado neste repositório. A proposta é simples — e um pouco teimosa: começar pequeno, documentar a origem do conhecimento e evoluir de forma comunitária sem cobrar tokens de API dos usuários.

> A Rubituci ainda é experimental. Ela não sabe tudo e o modelo atual é pequeno. Quando a geração bruta não atinge o nível mínimo de clareza, uma camada local de qualidade evita texto quebrado e pede uma fonte confiável. Isso é mais honesto do que improvisar bobagem com pose de especialista.

## Experimente

- Aplicação pública: [rubituci.com.br](https://rubituci.com.br)
- Site anteriormente hospedado na Rubituci: [nexusdc.com.br](https://nexusdc.com.br)
- Bugs e ideias: [GitHub Issues](https://github.com/Rubituci5/Rubituci-IA/issues)

## Propósito

1. **Modelo próprio:** arquitetura, tokenizador, pesos e treinamento controlados pelo projeto.
2. **Acesso aberto:** usuários não compram tokens de uma API proprietária. Hospedagem, CPU, memória e energia continuam tendo custo real.
3. **Aprendizado responsável:** conversas e páginas podem virar evidência, mas conteúdo novo entra em quarentena antes de afetar pesos de produção.
4. **Memória com origem:** fatos, fontes, confiança e contradições são registrados para auditoria.
5. **Evolução mensurável:** uma geração só substitui outra após avaliações e promoção explícita.

## Como foi concebida tecnicamente

```text
Navegador
   ├── Next.js — chat, cadastro/login e painel administrativo
   └── FastAPI
         ├── autenticação local + Google OAuth 2.0
         ├── Transformer Rubituci + tokenizador BPE próprios
         ├── filtro local de clareza em português
         ├── PostgreSQL + pgvector — usuários, sessões e memórias
         ├── pesquisa web com proveniência
         └── Celery + Redis — reflexão e sono computacional
```

O núcleo em `brain/` usa atenção causal, RoPE, RMSNorm e blocos SwiGLU. O vocabulário BPE é treinado do zero e os checkpoints ficam versionados por geração. Não há chamada a OpenAI, Gemini, Claude, Qwen ou outro modelo para produzir as respostas do chat.

`brain/language.py` normaliza a saída e mede sinais básicos de clareza. Se a geração tiver fragmentos, repetição ou pouca cobertura de português, ela é recusada e substituída por uma resposta local coerente. É uma proteção de produto, não uma alegação de fluência do checkpoint.

### Memória e aprendizado

- **Episódica:** registra conversas e eventos com data, origem, importância e confiança.
- **Semântica:** consolida conceitos recorrentes e relaciona evidências.
- **Crenças:** representa proposições, confiança e contradições.
- **Pesquisa:** coleta páginas com limites, fonte e histórico de consulta.
- **Candidatos de treino:** conhecimento novo é separado dos pesos ativos até revisão e avaliação.

### Sono computacional

Um processo agendado consolida experiências, identifica temas e dúvidas, pesquisa um número limitado de questões e produz candidatos de conhecimento. O sono não promove pesos automaticamente: uma geração candidata precisa superar avaliações de qualidade e segurança.

### Personalidade

A voz pretendida é jovem, informal e bem-humorada, com sarcasmo leve e ácido sem atacar pessoas. Quando não sabe, a Rubituci deve dizer isso e pedir uma fonte confiável — humildade epistemológica também é inteligência, por incrível que pareça.

## Estado atual

- Modelo próprio com aproximadamente 6,4 milhões de parâmetros.
- Chat HTTP e WebSocket.
- Cadastro por e-mail, login, refresh token e Google OAuth.
- Painel administrativo com usuários, sessões e acessos diários, copiável e exportável em CSV.
- Memória persistente em PostgreSQL/pgvector.
- Pesquisa e sono computacional por Celery/Redis.
- Interface responsiva em português.
- Deploy de referência com Nginx e systemd em `deploy/`.

O modelo ainda não possui conhecimento geral amplo e sua geração crua pode falhar. O filtro de qualidade protege a conversa, mas não substitui mais dados, avaliações e treinamento. Veja [a arquitetura detalhada](docs/ARQUITETURA.md).

## Executar localmente

Requisitos: Python 3.11+, Node.js 20+, PostgreSQL 15+ com pgvector e Redis 7+.

```bash
git clone https://github.com/Rubituci5/Rubituci-IA.git
cd Rubituci-IA
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
alembic upgrade head
npm --prefix web install
npm --prefix web run build
```

Em terminais separados:

```bash
uvicorn api.main:app --reload --port 8000
npm --prefix web run dev
celery -A worker.celery_app worker --beat --loglevel=info
```

## Google OAuth

Crie um cliente OAuth “Aplicativo da Web” no Google Auth Platform. Para desenvolvimento:

```text
Origem JavaScript: http://localhost:3000
URI de redirecionamento: http://localhost:8000/api/auth/google/callback
```

Em produção, use o domínio público. A URI cadastrada no Google precisa ser exatamente igual a `GOOGLE_REDIRECT_URI`. Guarde `GOOGLE_CLIENT_SECRET` somente no `.env` do servidor.

## Endpoints principais

| Método | Rota | Uso |
|---|---|---|
| `POST` | `/api/auth/register` | Cadastro local |
| `POST` | `/api/auth/login` | Login local |
| `GET` | `/api/auth/google` | Iniciar login Google |
| `POST` | `/api/chat` | Conversar |
| `WS` | `/ws/chat/{id}` | Conversa em streaming |
| `GET` | `/api/admin/users` | Usuários e acessos diários |
| `POST` | `/api/admin/learning/sleep` | Consolidação manual |
| `GET` | `/health` | Saúde da API e banco |

## Segurança e privacidade

Senhas usam bcrypt; tokens de sessão são armazenados como hash. Segredos ficam em variáveis de ambiente, pesquisa tem limites e proveniência, e entradas externas são evidência não confiável. Uma instalação pública deve publicar política de privacidade e obter consentimento antes de usar conversas para treinamento.

Não envie chaves, senhas ou dados pessoais em Issues. Para vulnerabilidades, consulte [SECURITY.md](SECURITY.md).

## Contribuir

Issues e Pull Requests são bem-vindos. A branch principal não recebe edição pública direta: contribuições passam por fork, revisão e testes. Leia [CONTRIBUTING.md](CONTRIBUTING.md).

Conteúdo para aprendizado precisa informar fonte, licença e contexto. O fato de algo estar na internet não significa que possa ser copiado ou que seja verdadeiro. A internet já se esforça bastante para provar isso diariamente.

## Licença

Código sob a [Licença MIT](LICENSE). Pesos, datasets e conteúdos de terceiros podem ter licenças próprias em seus manifestos.
