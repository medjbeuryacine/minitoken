"""
Mémoire structurée (structured memory) : extraction de faits durables,
classification par scope/type, et identification par clé logique stable.

Après un échange (message utilisateur + réponse de l'agent), un appel LLM
léger extrait les faits à retenir dans user_memory, et classe chacun comme
"global" (vrai pour tous les agents) ou "agent_specific" (propre à l'agent
concerné par CETTE conversation), ainsi que par un type fixe (voir
USER_MEMORY_TYPES) qui sert à organiser l'affichage côté frontend en
sections claires et éditables (Profil / Préférences / Objectifs /
Projets / Faits). Le code résout ensuite "agent_specific" en le scope
réel (ex: "coach_ia"), à partir du contexte connu de l'appelant — le LLM
n'a pas besoin de connaître le nom exact de l'agent pour classer.

POINT mémoire source de vérité : chaque fait reçoit aussi une
"memory_key", un identifiant logique STABLE (ex: "bench_press_goal")
destiné à repérer qu'un nouveau fait MET À JOUR un fait déjà connu,
plutôt que d'en créer un doublon -- plus fiable qu'une simple
correspondance type+category, qui peut légèrement varier d'un appel à
l'autre selon la formulation du LLM. Pour maximiser la stabilité, le LLM
reçoit la liste des memory_key déjà connues de cet utilisateur (voir
existing_memory_keys), et doit réutiliser une clé existante si le
nouveau fait parle du même sujet, plutôt que d'en inventer une nouvelle.

Le prompt seul n'est PAS fiable à 100% sur un petit modèle (8B) : plus on
ajoute de règles au prompt, plus le modèle se perd et invente de nouveaux
types d'erreurs. La stratégie retenue : un prompt SIMPLE et stable, plus
des filtres déterministes en Python en complément (_is_likely_meta_fact,
la validation du type, le filet de repli sur category si memory_key est
absent), qui ne dépendent jamais de la fiabilité du modèle.
"""

import json
from dataclasses import dataclass

from minitoken.database.models import USER_MEMORY_TYPES
from minitoken.providers.base import ChatMessage, LLMProvider

_TYPES_LIST = ", ".join(f'"{t}"' for t in USER_MEMORY_TYPES)

_EXTRACTION_SYSTEM_PROMPT_TEMPLATE = f"""Tu analyses un échange entre un utilisateur \
et un assistant IA. Extrais uniquement les faits DURABLES et PERSONNELS \
que l'utilisateur affirme sur lui-même (prénom, préférences, objectifs, \
contraintes, contexte de vie stable).

N'extrais PAS un fait si le message est une question, une demande de \
conseil, ou une situation hypothétique — même si elle mentionne un \
chiffre, un tiers, ou un sujet personnel. Exemple : "Combien de fois par \
semaine dois-je m'entraîner ?" ne révèle rien sur l'utilisateur, c'est \
une question, pas une affirmation.

Si l'utilisateur exprime un refus ou une négation ("je ne veux pas...", \
"jamais de..."), garde ce sens négatif clairement dans le fait extrait.

Pour chaque fait, indique un scope :
- "global" : vrai pour l'utilisateur en général (prénom, préférences de communication, contexte de vie/travail stable).
- "agent_specific" : donnée propre au sujet traité dans cette conversation (objectifs, contraintes, historique liés à ce domaine).

Pour chaque fait, indique aussi un "type", qui doit être EXACTEMENT l'une \
de ces {len(USER_MEMORY_TYPES)} valeurs, jamais une autre : {_TYPES_LIST}.
- "profile" : identité stable (prénom, métier, langue, situation de vie).
- "preference" : façon dont l'utilisateur aime interagir (style de réponse, niveau de détail).
- "goal" : un objectif que l'utilisateur veut atteindre.
- "project" : ce sur quoi l'utilisateur travaille actuellement.
- "fact" : toute autre information durable qui ne rentre dans aucune des catégories ci-dessus (contraintes, antécédents, habitudes).

Pour chaque fait, indique aussi une "category" COURTE et STABLE (2-3 mots \
maximum, en minuscules, toujours la même formulation pour un même type \
de fait -- ex: toujours "prénom", jamais "nom" une fois et "identité" une \
autre fois). Cette category est OBLIGATOIRE, jamais omise.

Pour chaque fait, indique enfin une "memory_key" : un identifiant COURT, \
STABLE, en minuscules avec underscores (ex: "bench_press_goal", \
"first_name", "preferred_language"), qui identifie le SUJET précis du \
fait, indépendamment de sa formulation exacte. C'est la clé la plus \
importante : si l'utilisateur donne une nouvelle valeur pour un sujet \
déjà connu (ex: change son objectif, change son métier), la memory_key \
DOIT être EXACTEMENT identique à celle déjà utilisée pour ce sujet, \
même si la formulation du fait change complètement.
{{existing_keys_section}}
Exemples :
- "Mon prénom est Karim" -> {{{{"fact": "S'appelle Karim", "scope": "global", "type": "profile", "category": "prénom", "memory_key": "first_name"}}}}
- "Je préfère des réponses courtes" -> {{{{"fact": "Préfère réponses courtes", "scope": "global", "type": "preference", "category": "style de réponse", "memory_key": "response_style"}}}}
- "Mon objectif est 100kg au développé couché" -> {{{{"fact": "Objectif 100kg développé couché", "scope": "agent_specific", "type": "goal", "category": "objectif force", "memory_key": "bench_press_goal"}}}}
- "Je m'entraîne lundi, mercredi, vendredi" -> {{{{"fact": "S'entraîne lundi/mercredi/vendredi", "scope": "agent_specific", "type": "fact", "category": "jours d'entraînement", "memory_key": "training_days"}}}}
- "Je construis une app SaaS avec React et Python" -> {{{{"fact": "Construit une app SaaS avec React et Python", "scope": "global", "type": "project", "category": "projet en cours", "memory_key": "current_project"}}}}
- "Je ne veux pas faire de squat, opération au genou" -> [{{{{"fact": "Refuse de faire du squat", "scope": "agent_specific", "type": "fact", "category": "restriction exercice", "memory_key": "squat_restriction"}}}}, {{{{"fact": "A eu une opération au genou", "scope": "agent_specific", "type": "fact", "category": "antécédent médical", "memory_key": "knee_surgery"}}}}]
- "Quel est le meilleur cardio pour débutant ?" -> aucun fait (question générale)
- "Combien d'entraînements par semaine ?" -> aucun fait (question, pas affirmation)
- "Un ami s'entraîne 6 fois par semaine, c'est trop ?" -> aucun fait (parle d'un tiers)

Si aucun fait personnel n'est donné, réponds avec une liste vide.

Réponds UNIQUEMENT avec un JSON valide, sans texte autour, au format :
[{{{{"fact": "...", "scope": "global" | "agent_specific", "type": "...", "category": "...", "memory_key": "..."}}}}]
Le type, la category et la memory_key ne doivent JAMAIS être omis ou vides.
"""


@dataclass
class ExtractedFact:
    fact: str
    scope: str  # résolu : "global" ou le nom réel de l'agent
    type: str = "fact"
    category: str | None = None
    memory_key: str | None = None


def extract_facts(
    *,
    provider: LLMProvider,
    current_agent_scope: str,
    user_message: str,
    assistant_response: str,
    existing_memory_keys: list[str] | None = None,
) -> list[ExtractedFact]:
    """
    Extrait les faits durables d'un échange, et résout leur scope :
    - "global" reste "global"
    - "agent_specific" devient `current_agent_scope` (l'agent réellement
      concerné par cette conversation, connu de l'appelant, pas du LLM)

    existing_memory_keys (optionnel) : liste des memory_key déjà connues
    de cet utilisateur (tous scopes confondus), injectée dans le prompt
    pour que le LLM réutilise une clé existante plutôt que d'en inventer
    une nouvelle pour le même sujet. Passer None ou une liste vide si
    l'utilisateur n'a encore aucun fait connu.

    Retourne une liste vide si rien à extraire ou si la réponse du LLM
    n'a pas pu être parsée (on ignore silencieusement plutôt que de
    planter le flux principal — l'extraction est un mécanisme en tâche de
    fond, pas critique pour la réponse déjà envoyée à l'utilisateur).

    Un filtre déterministe (_is_likely_meta_fact) élimine en plus les
    faits qui décrivent le sujet d'une question plutôt qu'une vraie info
    sur l'utilisateur — un garde-fou technique qui ne dépend pas de la
    fiabilité du modèle. Le type est validé contre USER_MEMORY_TYPES —
    jamais fait confiance aveuglément au LLM, même avec le prompt à jour.
    memory_key retombe sur un filet de repli déterministe (basé sur
    category) si le LLM l'omet malgré tout.
    """
    if existing_memory_keys:
        keys_list = ", ".join(f'"{k}"' for k in existing_memory_keys)
        existing_keys_section = (
            f"\nClés déjà connues pour cet utilisateur, à réutiliser si le "
            f"nouveau fait parle du même sujet : {keys_list}.\n"
        )
    else:
        existing_keys_section = ""

    system_prompt = _EXTRACTION_SYSTEM_PROMPT_TEMPLATE.format(
        existing_keys_section=existing_keys_section
    )

    user_content = (
        f"Message utilisateur :\n{user_message}\n\n"
        f"Réponse de l'assistant :\n{assistant_response}"
    )

    result = provider.generate(
        messages=[
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_content),
        ],
        max_tokens=400,
    )

    raw_facts = _parse_llm_json(result.content)
    if raw_facts is None:
        return []

    resolved_facts: list[ExtractedFact] = []
    for item in raw_facts:
        try:
            fact_text = item["fact"]
            raw_scope = item["scope"]
        except (KeyError, TypeError):
            continue  # entrée mal formée, on l'ignore plutôt que de planter

        if not fact_text or not isinstance(fact_text, str) or not fact_text.strip():
            continue  # fact vide/None/pas une string, on l'ignore

        if _is_likely_meta_fact(fact_text):
            continue  # filtre déterministe : ce n'est pas un vrai fait personnel

        resolved_scope = "global" if raw_scope == "global" else current_agent_scope

        raw_type = item.get("type")
        if not raw_type or not isinstance(raw_type, str) or raw_type.strip().lower() not in USER_MEMORY_TYPES:
            # Filet de sécurité déterministe : si le LLM renvoie un type
            # absent, vide, ou inventé (hors des 5 valeurs autorisées),
            # on retombe sur "fact" -- la valeur la plus générique,
            # jamais une exception qui casserait tout le flux d'extraction.
            fact_type = "fact"
        else:
            fact_type = raw_type.strip().lower()

        category = item.get("category")
        if not category or not isinstance(category, str) or not category.strip():
            # Filet de sécurité déterministe : si le LLM omet category
            # malgré l'instruction et les exemples, on en déduit une
            # valeur de repli grossière à partir du fait lui-même --
            # imparfait, mais mieux qu'une déduplication totalement
            # inactive.
            category = " ".join(fact_text.lower().split()[:4])
        category = category.strip().lower()

        memory_key = item.get("memory_key")
        if not memory_key or not isinstance(memory_key, str) or not memory_key.strip():
            # Filet de sécurité déterministe : si le LLM omet memory_key
            # malgré l'instruction et les exemples, on retombe sur
            # type+category comme clé de repli -- moins stable qu'une
            # vraie memory_key réfléchie par le LLM, mais toujours mieux
            # qu'aucune déduplication du tout.
            memory_key = f"{fact_type}_{category}".replace(" ", "_")
        else:
            memory_key = memory_key.strip().lower().replace(" ", "_")

        resolved_facts.append(
            ExtractedFact(
                fact=fact_text,
                scope=resolved_scope,
                type=fact_type,
                category=category,
                memory_key=memory_key,
            )
        )

    return resolved_facts


def _parse_llm_json(raw_content: str) -> list[dict] | None:
    """
    Parse la réponse JSON du LLM en tolérant les erreurs de format
    (le LLM peut parfois entourer le JSON de texte ou de balises markdown
    malgré l'instruction). Retourne None si le parsing échoue vraiment.
    """
    cleaned = raw_content.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, list):
        return None

    return parsed


def _is_likely_meta_fact(fact_text: str) -> bool:
    """
    Détecte les faits qui parlent de la question elle-même plutôt que
    d'une vraie info sur l'utilisateur — filet de sécurité déterministe,
    en complément du prompt (qui seul n'est pas fiable à 100% sur un
    petit modèle). Ce filtre ne dépend jamais du comportement du LLM,
    contrairement aux instructions du prompt qu'il peut ignorer.
    """
    meta_patterns = [
        "question sur", "pose une question", "s'intéresse à",
        "demande des infos", "se renseigne", "n'a pas donné",
        "aucune information", "pas d'information", "demande si",
        "veut savoir",
    ]
    fact_lower = fact_text.lower()
    return any(pattern in fact_lower for pattern in meta_patterns)