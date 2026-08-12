# minitoken

Mémoire conversationnelle **multi-provider, multi-user, multi-agent**, avec gestion de budget de tokens et rate limiting — légère, self-hosted, sur PostgreSQL + pgvector.

minitoken réduit le nombre de tokens envoyés à un LLM à chaque appel, sans perdre le sens de la conversation, en combinant 4 niveaux de mémoire, et protège vos appels contre les rate limits des fournisseurs (Groq, NVIDIA, OpenAI, ou tout autre provider OpenAI-compatible).

## Les 4 niveaux de mémoire

- **Short-term** — les derniers messages, envoyés en clair
- **Summary** — un résumé compact de tout ce qui précède, mis à jour de façon incrémentale
- **Structured** — des faits durables extraits automatiquement, classés par *scope* (`global` ou propre à un agent précis)
- **Vector** — une recherche sémantique dans les anciens échanges, pour retrouver un sujet mentionné il y a longtemps

## Pourquoi minitoken

- **Multi-provider** : Groq, OpenAI, NVIDIA (compatibles OpenAI) et Anthropic (Claude), via une interface commune — aucune liste fermée, indiquez simplement `base_url`
- **Multi-user** : chaque utilisateur a sa mémoire isolée, jamais de fuite entre comptes
- **Multi-agent** : un même utilisateur peut parler à plusieurs agents ; chaque fait retenu est soit `global` (visible par tous les agents), soit propre à un agent précis
- **Rate limiting intégré** : RPM/RPD/TPM/TPD par provider, avec refus immédiat + délai (jamais d'attente silencieuse), et lecture des vrais headers du fournisseur quand disponibles (ex: Groq)
- **Léger** : uniquement PostgreSQL + pgvector, aucune dépendance lourde
- **Gratuit par défaut** : embeddings calculés localement (`sentence-transformers`), aucun appel API payant requis pour la mémoire vectorielle

## Pré-requis

minitoken s'installe **dans un projet existant**. Il a besoin de deux tables déjà présentes dans votre base de données :

- une table **users** (au minimum une clé primaire)
- une table **conversations** (au minimum une clé primaire)

minitoken ne crée jamais ces deux tables — il les référence par clé étrangère (`ON DELETE CASCADE`/`SET NULL` selon la table). Si vous n'en avez pas encore :

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid()
);

CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id)
);
```

## Installation

```bash
pip install "minitoken[openai-compatible,local-embeddings,huggingface-tokenizers] @ git+https://github.com/medjbeuryacine/minitoken.git"
```

Groupes optionnels disponibles : `openai-compatible` (Groq/OpenAI/NVIDIA), `anthropic` (Claude), `local-embeddings` (sentence-transformers), `huggingface-tokenizers` (comptage précis), `prompt-compression` (LLMLingua-2), `all`.

## Démarrage rapide

```python
from minitoken.config import MinitokenConfig
from minitoken.client import MinitokenClient
from minitoken.providers import ChatMessage
from minitoken.token_budget.rate_limiter import RateLimitConfig

config = MinitokenConfig(
    database_url="postgresql://user:password@localhost:5432/votre_db",
    users_table="users", users_id_column="id",
    conversations_table="conversations", conversations_id_column="id",

    # Provider utilisé pour compter les tokens (doit matcher le modèle
    # réellement utilisé pour vos réponses, généré en dehors de minitoken)
    token_counter_provider="groq",
    token_counter_model="llama-3.3-70b-versatile",
    token_counter_api_key="gsk_...",
    token_counter_base_url="https://api.groq.com/openai/v1",
    token_counter_rate_limit=RateLimitConfig(requests_per_minute=30, tokens_per_minute=12000),

    # Provider pour le résumé et l'extraction de faits (souvent plus léger)
    extraction_provider="groq",
    extraction_model="llama-3.1-8b-instant",
    extraction_api_key="gsk_...",
    extraction_base_url="https://api.groq.com/openai/v1",
    extraction_rate_limit=RateLimitConfig(requests_per_minute=28, tokens_per_minute=5500),

    embedding_provider="local",       # gratuit, sentence-transformers
    token_budget_max=8000,
    max_user_facts=30,                # limite les faits récupérés par appel, peu importe le volume accumulé
    agent_scopes=["coach", "programme"],
)

client = MinitokenClient(config)
client.initialize()  # crée les tables, active pgvector — une seule fois
```

### À chaque message reçu

```python
# 1. Avant de générer la réponse : récupérer le contexte optimisé
bundle, token_report = client.get_context(
    conversation_id=conversation_id,
    user_id=user_id,
    agent_scope="coach",
    all_messages=all_messages,
)

# 2. Si votre LLM de réponse est appelé en dehors de minitoken (ex: via
#    LangChain), vérifiez le rate limit vous-même avant l'appel réel :
rate_check = client.check_response_rate_limit(estimated_tokens=token_report.total_tokens)
if not rate_check.allowed:
    # renvoyez une erreur claire avec rate_check.retry_after_seconds,
    # ne bloquez jamais en silence
    ...

# 3. Envoyez bundle.recent_messages + bundle.summary + bundle.structured_facts
#    + bundle.vector_memories à votre LLM.

# 4. Après la réponse, en tâche de fond (non bloquant) :
client.record_exchange(
    conversation_id=conversation_id,
    user_id=user_id,
    agent_scope="coach",
    all_messages=all_messages,
    user_message=derniere_question,
    assistant_response=reponse_generee,
)

# 5. Optionnel : état du quota pour affichage (barre de progression)
usage = client.get_rate_limit_status(provider_role="token_counter")
```

## Le concept de `scope`

Chaque fait retenu dans `user_memory` a un `scope` :

- `"global"` — vrai peu importe l'agent (préférences de communication, contexte stable)
- le nom d'un agent (ex: `"coach"`, `"programme"`) — pertinent uniquement pour cet agent

`get_context()` ne récupère que les faits `global` + ceux du scope de l'agent courant — jamais ceux d'un autre agent. La classification est faite automatiquement par le LLM d'extraction à chaque échange, guidée par un prompt testé sur des cas pièges (négations, questions hypothétiques, mentions de tiers) pour limiter les hallucinations d'un petit modèle.

## Rate limiting

Configurable indépendamment pour chaque provider (`token_counter`, `extraction`, `compression`), sur 4 dimensions optionnelles :

```python
RateLimitConfig(
    requests_per_minute=30,
    requests_per_day=14400,
    tokens_per_minute=6000,
    tokens_per_day=500000,
)
```

Deux méthodes disponibles selon le contexte d'usage :

- **`check_limit()`** — ne bloque jamais, retourne immédiatement `allowed=False` + `retry_after_seconds` si la limite serait dépassée. À utiliser dans un chemin visible par l'utilisateur (via `check_response_rate_limit()` côté client).
- **`wait_if_needed()`** — attend silencieusement le temps nécessaire avant de laisser passer l'appel. Adapté à un usage en tâche de fond (ex: extraction), où l'attente est invisible pour l'utilisateur final.

L'historique des appels est stocké dans une table Postgres partagée (`rate_limit_events`), pas en mémoire du process — fonctionne correctement même avec plusieurs instances/workers de l'application hôte.

Si le fournisseur expose de vrais headers de rate limit (`x-ratelimit-*`, format Groq), `get_rate_limit_status()` les utilise en priorité, plus précis que le calcul interne.

## Compression de texte (optionnel, désactivé par défaut)

```python
compression_mode="local"   # LLMLingua-2, gratuit, extractif — ou "api" pour un LLM génératif
compression_target_ratio=0.6
```

⚠️ Testé et **désactivé par défaut** (`compression_mode="none"`) : peut perdre des négations sur certaines formulations. À activer avec prudence, en testant sur votre propre contenu avant la mise en production.

## Architecture interne

```
minitoken/
├── config.py                # MinitokenConfig
├── client.py                 # MinitokenClient — point d'entrée public
├── providers/                 # adaptateurs LLM (OpenAI-compatible + Anthropic)
├── database/
│   ├── models.py                # définition des 4 tables
│   ├── repository.py            # lecture/écriture, filtrage par scope
│   └── migrate.py               # création tables + activation pgvector (programmatique)
├── memory/                    # short_term, summary, structured, vector, prompt_compression
└── token_budget/
    ├── counter.py                # comptage de tokens du bundle assemblé
    ├── trimmer.py                # réduction si dépassement du budget
    └── rate_limiter.py           # RPM/RPD/TPM/TPD, fenêtre glissante Postgres
```

### Pourquoi pas de fichiers Alembic classiques

La structure réelle dépend de la config du développeur (noms de tables `users`/`conversations`, dimension du vecteur selon l'embedder). `migrate.py` applique donc la migration de façon programmatique, à partir de la config réelle fournie par `MinitokenClient`.

### Notes de migration de schéma

Modifier `models.py` (ex: une contrainte `ON DELETE CASCADE`) ne met **pas** à jour les tables déjà créées — `create_all()` ne touche que les tables absentes. Il faut corriger manuellement en SQL ou recréer les tables.

## Licence

MIT