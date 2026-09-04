# Aprendizado comunitário responsável

Rubituci pode receber conhecimento da comunidade, mas uma contribuição não se
torna verdade nem deve alterar os pesos imediatamente. O fluxo recomendado é:

1. receber texto, licença, autor, idioma e URL de origem;
2. remover dados pessoais, segredos, spam, duplicatas e instruções maliciosas;
3. colocar a contribuição em quarentena;
4. obter revisão independente e registrar divergências;
5. separar conjuntos de treino, validação e teste antes do treinamento;
6. treinar uma candidata em snapshot novo, nunca sobre o modelo em produção;
7. avaliar português, factualidade, segurança, memorização e regressões;
8. promover somente uma versão aprovada, mantendo rollback e proveniência.

Conteúdo da web é evidência não confiável. Páginas podem conter erros, propaganda
ou *prompt injection*. A navegação deve respeitar `robots.txt`, termos de uso,
direitos autorais, limites por domínio, bloqueio de redes privadas e um orçamento
de requisições. A entidade pode formular perguntas e sugerir fontes com
curiosidade; permissões operacionais e critérios de promoção continuam externos
ao modelo.

## Critérios mínimos de alfabetização

- preservação correta de UTF-8, acentos e pontuação;
- frases completas e sem repetição degenerativa;
- concordância nominal e verbal em tarefas controladas;
- compreensão, resumo, correção e resposta curta em português brasileiro;
- capacidade de declarar incerteza em vez de inventar;
- conjunto de teste congelado e nunca usado no treino.

O arquivo `data/literacy_ptbr_v1/literacy_ptbr_v1.jsonl` é apenas a primeira
semente. Ele é pequeno demais para alfabetizar sozinho um transformer treinado do
zero; serve para tornar o objetivo mensurável e iniciar contribuições revisáveis.

## Independência do modelo

O backend oficial usa exclusivamente `EntityTransformer` e `BPETokenizer`, ambos
implementados neste repositório. Pesos, tokenizadores ou respostas de modelos
externos não entram no caminho de produção. Experimentos históricos devem ficar
claramente separados e nunca ser promovidos como gerações da Rubituci.
