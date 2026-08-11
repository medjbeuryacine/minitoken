"""
Mémoire structurée (structured memory) : extraction de faits durables +
classification par scope.

Après un échange (message utilisateur + réponse de l'agent), un appel LLM
léger extrait les faits à retenir dans user_memory, et classe chacun comme
"global" (vrai pour tous les agents) ou "agent_specific" (propre à l'agent
concerné par CETTE conversation). Le code résout ensuite "agent_specific"
en le scope réel (ex: "coach_ia"), à partir du contexte connu de l'appelant
— le LLM n'a pas besoin de connaître le nom exact de l'agent pour classer.
"""

import json
from dataclasses import dataclass

from minitoken.providers.base import ChatMessage, LLMProvider

_EXTRACTION_SYSTEM_PROMPT = """Tu analyses un échange entre un utilisateur \
et un assistant IA. Ton rôle est d'être TRÈS SÉLECTIF — la plupart des \
échanges ne contiennent AUCUN fait à extraire, et c'est normal.

RÈGLE PRINCIPALE : n'extrais un fait QUE si l'utilisateur donne une \
information PERSONNELLE, EXPLICITE et STABLE sur lui-même (pas sur le \
sujet en général). Une question posée par l'utilisateur n'est PAS un \
fait sur lui, même si elle mentionne un mot-clé personnel.

Pour chaque fait retenu, indique un scope :
- "global" si le fait concerne l'utilisateur en général, INDÉPENDAMMENT du \
sujet de la conversation (préférences de communication, contexte de \
vie/travail stable).
- "agent_specific" si le fait est une DONNÉE personnelle propre au sujet \
traité (objectifs, contraintes, historique propres à ce domaine).

Exemples de faits À EXTRAIRE (information personnelle explicite donnée par
l'utilisateur) :
- "Je préfère des réponses courtes" -> global
- "Mon objectif est 100kg au développé couché d'ici 1 an" -> agent_specific
- "Je m'entraîne lundi, mercredi et vendredi" -> agent_specific
- "J'ai mal au genou droit depuis 2 semaines" -> agent_specific

Exemples de faits à NE PAS EXTRAIRE (aucune information personnelle réelle,
juste une question ou une remarque générale) :
- "Quel est le meilleur cardio pour un débutant ?" -> RIEN (question générale, pas d'info sur cet utilisateur précis)
- "Comment optimiser ma récupération ?" -> RIEN (question générale)
- "Quels suppléments prennent les athlètes professionnels ?" -> RIEN (parle des autres, pas de l'utilisateur)
- "Comment faire des étirements après une séance ?" -> RIEN (question générale de méthode)
- La réponse de l'assistant seule ne donne JAMAIS lieu à un fait — seul ce
  que L'UTILISATEUR affirme sur lui-même compte.

En cas de doute, NE RIEN EXTRAIRE. Il vaut mieux rater un fait mineur que
polluer la mémoire avec du bruit. La très grande majorité des échanges
(questions générales, demandes de conseils, échanges de politesse) ne
contiennent RIEN à extraire — réponds alors avec une liste vide.

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