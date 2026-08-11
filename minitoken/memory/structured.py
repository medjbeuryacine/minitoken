"""
Mémoire structurée (structured memory) : extraction de faits durables +
classification par scope.

Après un échange (message utilisateur + réponse de l'agent), un appel LLM
léger extrait les faits à retenir dans user_memory, et classe chacun comme
"global" (vrai pour tous les agents) ou "agent_specific" (propre à l'agent
concerné par CETTE conversation). Le code résout ensuite "agent_specific"
en le scope réel (ex: "coach_ia"), à partir du contexte connu de l'appelant
— le LLM n'a pas besoin de connaître le nom exact de l'agent pour classer.

Le prompt seul n'est PAS fiable à 100% sur un petit modèle (8B) : plus on
ajoute de règles au prompt, plus le modèle se perd et invente de nouveaux
types d'erreurs. La stratégie retenue : un prompt SIMPLE et stable, plus
un filtre déterministe en Python en complément (_is_likely_meta_fact),
qui ne dépend jamais de la fiabilité du modèle.
"""

import json
from dataclasses import dataclass

from minitoken.providers.base import ChatMessage, LLMProvider

_EXTRACTION_SYSTEM_PROMPT = """Tu analyses un échange entre un utilisateur \
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

Exemples :
- "Mon prénom est Karim" -> {"fact": "S'appelle Karim", "scope": "global"}
- "Je préfère des réponses courtes" -> {"fact": "Préfère réponses courtes", "scope": "global"}
- "Mon objectif est 100kg au développé couché" -> {"fact": "Objectif 100kg développé couché", "scope": "agent_specific"}
- "Je m'entraîne lundi, mercredi, vendredi" -> {"fact": "S'entraîne lundi/mercredi/vendredi", "scope": "agent_specific"}
- "Je ne veux pas faire de squat, opération au genou" -> [{"fact": "Refuse de faire du squat", "scope": "agent_specific"}, {"fact": "A eu une opération au genou", "scope": "agent_specific"}]
- "Quel est le meilleur cardio pour débutant ?" -> aucun fait (question générale)
- "Combien d'entraînements par semaine ?" -> aucun fait (question, pas affirmation)
- "Un ami s'entraîne 6 fois par semaine, c'est trop ?" -> aucun fait (parle d'un tiers)

Si aucun fait personnel n'est donné, réponds avec une liste vide.

Réponds UNIQUEMENT avec un JSON valide, sans texte autour, au format :
[{"fact": "...", "scope": "global" | "agent_specific", "category": "..."}]
"""


@dataclass
class ExtractedFact:
    fact: str
    scope: str  # résolu : "global" ou le nom réel de l'agent
    category: str | None = None


def extract_facts(
    *,
    provider: LLMProvider,
    current_agent_scope: str,
    user_message: str,
    assistant_response: str,
) -> list[ExtractedFact]:
    """
    Extrait les faits durables d'un échange, et résout leur scope :
    - "global" reste "global"
    - "agent_specific" devient `current_agent_scope` (l'agent réellement
      concerné par cette conversation, connu de l'appelant, pas du LLM)

    Retourne une liste vide si rien à extraire ou si la réponse du LLM
    n'a pas pu être parsée (on ignore silencieusement plutôt que de
    planter le flux principal — l'extraction est un mécanisme en tâche de
    fond, pas critique pour la réponse déjà envoyée à l'utilisateur).

    Un filtre déterministe (_is_likely_meta_fact) élimine en plus les
    faits qui décrivent le sujet d'une question plutôt qu'une vraie info
    sur l'utilisateur — un garde-fou technique qui ne dépend pas de la
    fiabilité du modèle.
    """
    user_content = (
        f"Message utilisateur :\n{user_message}\n\n"
        f"Réponse de l'assistant :\n{assistant_response}"
    )

    result = provider.generate(
        messages=[
            ChatMessage(role="system", content=_EXTRACTION_SYSTEM_PROMPT),
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

        resolved_facts.append(
            ExtractedFact(
                fact=fact_text,
                scope=resolved_scope,
                category=item.get("category"),
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

    # Tolère un éventuel wrapping en balises markdown ```json ... ```
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