# minitoken

Mémoire conversationnelle **multi-provider, multi-user, multi-agent**, avec gestion de budget de tokens — légère, self-hosted, sur PostgreSQL + pgvector.

minitoken réduit le nombre de tokens envoyés à un LLM à chaque appel, sans perdre le sens de la conversation, en combinant 4 niveaux de mémoire :

- **Short-term** — les derniers messages, envoyés en clair
- **Summary** — un résumé compact de tout ce qui précède, mis à jour en continu
- **Structured** — des faits durables extraits automatiquement, classés par *scope* (`global` ou propre à un agent précis)
- **Vector** — une recherche sémantique dans les anciens échanges, pour retrouver un sujet mentionné il y a longtemps

## Pourquoi minitoken

- **Multi-provider** : Groq, OpenAI, NVIDIA (compatibles OpenAI) et Anthropic (Claude), via une interface commune
- **Multi-user** : chaque utilisateur a sa mémoire isolée, jamais de fuite entre comptes
- **Multi-agent** : un même utilisateur peut parler à plusieurs agents (ex: un coach général et un agent avec accès à des données personnelles) ; chaque fait retenu est soit `global` (visible par tous les agents), soit propre à un agent précis
- **Léger** : uniquement PostgreSQL + pgvector, aucune dépendance lourde (pas de Neo4j, pas de service tiers obligatoire)
- **Gratuit par défaut** : embeddings calculés localement (`sentence-transformers`), aucun appel API payant requis pour la mémoire vectorielle

## Pré-requis

minitoken s'installe **dans un projet existant**. Il a besoin de deux tables déjà présentes dans votre base de données :

- une table **users** (au minimum une clé primaire)
- une table **conversations** (au minimum une clé primaire)

minitoken ne crée jamais ces deux tables — il les référence par clé étrangère. Si vous n'en avez pas encore, voici un schéma minimal :

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid()
);

CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id)
);
```

Votre base PostgreSQL doit aussi pouvoir installer l'extension `pgvector` (minitoken l'active automatiquement lors de l'initialisation, si l'extension est disponible sur votre instance Postgres).

## Installation

```bash
# Avec Groq / OpenAI / NVIDIA + embeddings locaux (recommandé pour démarrer)
pip install "minitoken[openai-compatible,local-embeddings,huggingface-tokenizers] @ git+https://github.com/medjbeuryacine/minitoken.git"

# Avec Claude (Anthropic)
pip install "minitoken[anthropic,local-embeddings] @ git+https://github.com/medjbeuryacine/minitoken.git"

# Tout installer
pip install "minitoken[all] @ git+https://github.com/medjbeuryacine/minitoken.git"
```

## Démarrage rapide

```python
from minitoken.config import MinitokenConfig
from minitoken.client import MinitokenClient
from minitoken.providers import ChatMessage

config = MinitokenConfig(
    database_url="postgresql://user:password@localhost:5432/votre_db",

    users_table="users",
    users_id_column="id",

    conversations_table="conversations",
    conversations_id_column="id",

    response_provider="groq",
    response_model="llama-3.3-70b-versatile",
    response_api_key="gsk_...",

    extraction_provider="groq",
    extraction_model="llama-3.1-8b-instant",  # modèle plus léger pour le résumé/l'extraction
    extraction_api_key="gsk_...",

    embedding_provider="local",  # gratuit, aucune clé requise
    token_budget_max=8000,

    agent_scopes=["coach_ia", "mon_programme"],  # vos agents ; "global" est ajouté automatiquement
)

client = MinitokenClient(config)
client.initialize()  # à exécuter une seule fois : crée les tables + active pgvector
```

### À chaque message reçu

```python
# 1. Avant de générer la réponse : récupérer le contexte optimisé
bundle, token_report = client.get_context(
    conversation_id=conversation_id,
    user_id=user_id,
    agent_scope="coach_ia",
    all_messages=all_messages,  # historique complet de la conversation, en ChatMessage
)

print(token_report.total_tokens)  # toujours <= token_budget_max

# 2. Envoyer bundle.recent_messages + bundle.summary + bundle.structured_facts
#    + bundle.vector_memories à votre LLM, selon le format de votre prompt.

# 3. Après avoir envoyé la réponse à l'utilisateur (idéalement en tâche de
#    fond, non bloquant) : mettre à jour la mémoire
client.record_exchange(
    conversation_id=conversation_id,
    user_id=user_id,
    agent_scope="coach_ia",
    all_messages=all_messages,
    user_message=derniere_question_utilisateur,
    assistant_response=reponse_generee,
)
```

## Le concept de `scope`

Chaque fait retenu dans `user_memory` (et chaque souvenir vectorisé) a un `scope` :

- `"global"` — vrai peu importe l'agent auquel l'utilisateur s'adresse (préférences de communication, contexte stable)
- le nom d'un agent (ex: `"coach_ia"`, `"mon_programme"`) — pertinent uniquement pour cet agent précis

Quand un agent appelle `get_context()`, il ne récupère que les faits `global` + ceux de son propre scope — jamais ceux d'un autre agent. La classification `global` / spécifique est faite automatiquement par un LLM léger à chaque échange, guidé par des exemples dans son prompt pour rester fiable même sur de petits modèles.

## Architecture interne

```text
minitoken/
├── config.py                 # MinitokenConfig — toute la configuration du développeur
├── client.py                 # MinitokenClient — point d'entrée public
├── providers/                # Adaptateurs LLM (Groq/OpenAI/NVIDIA + Anthropic)
├── database/
│   ├── models.py             # Définition des 3 tables
│   ├── repository.py         # Lecture/écriture, filtrage par scope
│   └── migrate.py            # Création des tables + activation de pgvector
│                             # (programmatiquement, sans Alembic)
├── memory/                   # Les 4 niveaux : short_term, summary, structured, vector
└── token_budget/             # Comptage + réduction si dépassement du budget
```

### Pourquoi pas de fichiers Alembic classiques

La structure réelle des tables dépend de la config du développeur (noms de tables `users`/`conversations`, dimension du vecteur selon l'embedder choisi) — des fichiers de migration figés à l'avance ne pourraient pas s'adapter à chaque projet différent.

`migrate.py` applique donc la migration de façon programmatique, à partir de la config réelle fournie par `MinitokenClient`.

## Licence

MIT