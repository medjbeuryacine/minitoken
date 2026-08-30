"""
Réduction automatique du contexte si le budget de tokens est dépassé.

Ordre de priorité de coupe (du premier coupé au dernier, comme décidé) :
  1. vector_memories     (le moins critique — souvenirs anciens, "bonus")
  2. structured_facts    (les plus anciens/moins pertinents d'abord)
  3. summary              (recompression, jamais suppression totale)
  4. recent_messages     (jamais coupé)
  4. conversation_state  (jamais coupé, même priorité que recent_messages
                           — état structuré de la tâche en cours, sa perte
                           provoquerait exactement le bug qu'il existe
                           pour corriger : une valeur oubliée entre deux
                           tours de clarification)
"""

from minitoken.providers.base import LLMProvider
from minitoken.token_budget.counter import ContextBundle, TokenReport, count_bundle_tokens


def trim_to_budget(
    *,
    provider: LLMProvider,
    bundle: ContextBundle,
    budget_max: int,
) -> tuple[ContextBundle, TokenReport]:
    """
    Réduit le bundle jusqu'à rentrer dans le budget, en coupant dans
    l'ordre de priorité défini ci-dessus. Retourne le bundle réduit et le
    rapport de tokens final.

    Ne lève jamais d'exception si le budget ne peut pas être atteint même
    après avoir tout coupé sauf recent_messages : dans ce cas de figure
    extrême (recent_messages seul dépasse déjà le budget), retourne le
    bundle réduit au minimum avec un dépassement encore présent — c'est à
    l'appelant de décider quoi faire (ex: réduire keep_recent_count en
    amont, dans short_term.py).
    """
    working_bundle = ContextBundle(
        recent_messages=list(bundle.recent_messages),
        summary=bundle.summary,
        structured_facts=list(bundle.structured_facts),
        vector_memories=list(bundle.vector_memories),
        conversation_state=dict(bundle.conversation_state),
    )

    report = count_bundle_tokens(provider=provider, bundle=working_bundle)
    if report.total_tokens <= budget_max:
        return working_bundle, report

    # 1. Coupe les vector_memories, du dernier (moins pertinent, résultats
    #    de recherche généralement déjà triés par pertinence décroissante)
    #    au premier, jusqu'à rentrer dans le budget ou les épuiser.
    while working_bundle.vector_memories and report.total_tokens > budget_max:
        working_bundle.vector_memories.pop()
        report = count_bundle_tokens(provider=provider, bundle=working_bundle)

    if report.total_tokens <= budget_max:
        return working_bundle, report

    # 2. Coupe les structured_facts les plus anciens en premier (on
    #    suppose la liste déjà triée du plus récent au plus ancien, comme
    #    le fait repository.get_user_facts() avec son order_by).
    while working_bundle.structured_facts and report.total_tokens > budget_max:
        working_bundle.structured_facts.pop()
        report = count_bundle_tokens(provider=provider, bundle=working_bundle)

    if report.total_tokens <= budget_max:
        return working_bundle, report

    # 3. Recompresse le summary plutôt que de le supprimer entièrement —
    #    ici on tronque en dernier recours si aucun LLM de recompression
    #    n'est disponible dans ce contexte (le trimmer est volontairement
    #    synchrone/sans appel LLM ; la vraie recompression via LLM se fait
    #    en amont par memory/summary.py:recompress_if_too_long avant
    #    d'arriver ici).
    #
    #    Garde-fou anti-boucle-infinie : _truncate_by_tokens converge vers
    #    un minimum de 1 caractère (max(1, ...)) et ne réduit plus jamais
    #    en dessous. Si ce caractère unique compte encore comme un ou
    #    plusieurs tokens dépassant un budget_max très petit, la boucle ne
    #    progresserait jamais sans cette protection — on sort dès que la
    #    troncature n'a plus réduit la longueur du texte.
    while working_bundle.summary and report.total_tokens > budget_max:
        previous_length = len(working_bundle.summary)
        working_bundle.summary = _truncate_by_tokens(
            provider=provider, text=working_bundle.summary, reduce_ratio=0.7
        )
        if len(working_bundle.summary) >= previous_length:
            # La troncature n'a plus d'effet (texte déjà au minimum) —
            # on arrête pour éviter une boucle infinie. Le dépassement
            # éventuel restant est signalé via le report retourné, comme
            # pour recent_messages (voir docstring de la fonction).
            break
        report = count_bundle_tokens(provider=provider, bundle=working_bundle)

    # 4. recent_messages n'est jamais coupé ici — s'il reste un
    #    dépassement à ce stade, c'est signalé à l'appelant via le report
    #    (report.total_tokens > budget_max), qui décide de la suite.
    return working_bundle, report


def _truncate_by_tokens(*, provider: LLMProvider, text: str, reduce_ratio: float) -> str:
    """Tronque grossièrement un texte à une fraction de sa taille
    actuelle. Utilisé uniquement en dernier recours (voir étape 3
    ci-dessus) — la vraie recompression intelligente passe par le LLM en
    amont, pas par cette troncature mécanique."""
    target_char_count = max(1, int(len(text) * reduce_ratio))
    return text[:target_char_count]