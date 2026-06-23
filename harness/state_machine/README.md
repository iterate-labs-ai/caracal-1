# State machine policy DSL

Estados rigidos para o modulo Crafter. Forca disciplina, evita chuteira chain-of-thought.

Tokens especiais delimitam estados: `[ANALYZE]`, `[HYPOTHESIZE]`, `[INSTRUMENT]`, `[EXECUTE]`, `[REFINE]`, `[PATCH]`.

Validator harness verifica transicoes em runtime. Reward parcial por progressao valida.

Mutavel via Loop D (scaffold mutation, classe B) com 1 review.
