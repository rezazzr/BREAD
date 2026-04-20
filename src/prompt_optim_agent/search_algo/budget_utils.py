"""Budget comparison utilities for TRAS vs APE.

Provides a function to compute comparable APE hyperparameters given TRAS settings,
ensuring both methods use a similar total computational budget.
"""


def compute_ape_budget_from_tras(
    iteration_num: int,
    expand_width: int,
    num_new_prompts: int = 1,
    gradient_sampling: int = 1,
    positive_reinforcement_depth: int = 0,
    depth_limit: int = 8,
    train_batch_size: int = 5,
    num_demos: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Compute comparable APE hyperparameters given TRAS settings.

    The goal is to equate the total number of candidate prompts evaluated on
    the eval set, since that is the dominant cost shared by both methods.

    Args:
        iteration_num: TRAS MCTS iterations
        expand_width: children generated per MCTS expansion
        num_new_prompts: prompts generated per gradient step in TRAS
        gradient_sampling: number of gradient samples aggregated in TRAS
        positive_reinforcement_depth: depth threshold for positive gradients in TRAS
        depth_limit: max MCTS tree depth
        train_batch_size: TRAS training batch size
        num_demos: number of demos per APE generation query (user preference)
        verbose: print detailed breakdown

    Returns:
        dict with recommended APE hyperparameters:
            - num_subsamples
            - num_demos
            - num_prompts_per_subsample
            - total_candidates
        and budget breakdown:
            - tras_total_nodes_evaluated
            - tras_optim_calls_per_expansion
            - tras_base_calls_per_expansion
    """

    # === TRAS budget analysis ===

    # Per expansion (one world_model.step call):
    #   1 base_model forward on training batch
    #   gradient_sampling optim_model calls for negative gradient
    #   1 optim_model call for negative gradient summary (if gradient_sampling > 1)
    #   gradient_sampling optim_model calls for positive gradient (conditional)
    #   1 optim_model call for positive gradient summary (conditional, if gradient_sampling > 1)
    #   1 optim_model call for optimize (generate new prompts)
    #   num_new_prompts base_model eval passes (one per child node)

    neg_gradient_calls = gradient_sampling
    neg_summary_calls = 1 if gradient_sampling > 1 else 0

    # Positive gradient fires when: depth >= positive_reinforcement_depth AND mixed accuracy
    # On average, roughly half the expansions will have mixed accuracy at sufficient depth.
    # Fraction of depths where positive gradient is active:
    active_depths = max(0, depth_limit - positive_reinforcement_depth)
    pos_fraction = active_depths / depth_limit if depth_limit > 0 else 0.5
    # Discount further by ~50% for mixed accuracy (not all-correct or all-wrong)
    pos_fraction *= 0.5

    pos_gradient_calls = gradient_sampling * pos_fraction
    pos_summary_calls = (1 if gradient_sampling > 1 else 0) * pos_fraction

    optim_calls_per_expansion = (
        neg_gradient_calls
        + neg_summary_calls
        + pos_gradient_calls
        + pos_summary_calls
        + 1  # optimize call
    )

    base_calls_per_expansion = 1 + num_new_prompts  # forward + eval per child

    # Total expansions: minimum is iteration_num * expand_width (one expansion per
    # iteration). Simulation may expand additional nodes, but this is hard to predict.
    # Use the minimum as a conservative estimate.
    total_expansions = iteration_num * expand_width

    # Total child nodes evaluated = primary budget metric
    tras_total_nodes_evaluated = total_expansions * num_new_prompts
    tras_total_optim_calls = total_expansions * optim_calls_per_expansion
    tras_total_base_calls = total_expansions * base_calls_per_expansion

    # === APE budget computation ===

    # Match total candidates evaluated to TRAS total nodes evaluated
    total_candidates = tras_total_nodes_evaluated

    # Distribute across subsamples and prompts_per_subsample
    # Heuristic: keep prompts_per_subsample moderate (3-5) for diversity
    num_prompts_per_subsample = min(5, max(1, total_candidates // 5))
    num_subsamples = max(1, total_candidates // num_prompts_per_subsample)

    # Adjust to match total exactly
    actual_total = num_subsamples * num_prompts_per_subsample
    if actual_total < total_candidates:
        num_subsamples += 1
        actual_total = num_subsamples * num_prompts_per_subsample

    ape_optim_calls = num_subsamples  # one generation call per subsample
    ape_base_eval_calls = actual_total  # one eval pass per candidate

    result = {
        # Recommended APE hyperparameters
        "num_subsamples": num_subsamples,
        "num_demos": num_demos,
        "num_prompts_per_subsample": num_prompts_per_subsample,
        "total_candidates": actual_total,
        # TRAS budget breakdown
        "tras_total_nodes_evaluated": tras_total_nodes_evaluated,
        "tras_total_expansions": total_expansions,
        "tras_optim_calls_per_expansion": round(optim_calls_per_expansion, 1),
        "tras_total_optim_calls": round(tras_total_optim_calls, 1),
        "tras_total_base_calls": tras_total_base_calls,
        # APE budget breakdown
        "ape_total_optim_calls": ape_optim_calls,
        "ape_total_base_eval_calls": ape_base_eval_calls,
    }

    if verbose:
        print("=" * 65)
        print("  TRAS vs APE Budget Comparison")
        print("=" * 65)
        print()
        print("TRAS Configuration:")
        print(f"  iteration_num             = {iteration_num}")
        print(f"  expand_width              = {expand_width}")
        print(f"  num_new_prompts           = {num_new_prompts}")
        print(f"  gradient_sampling         = {gradient_sampling}")
        print(f"  positive_reinforcement_depth = {positive_reinforcement_depth}")
        print(f"  depth_limit               = {depth_limit}")
        print()
        print("TRAS Budget (estimated):")
        print(f"  Total expansions          = {total_expansions}")
        print(f"  Nodes evaluated (children)= {tras_total_nodes_evaluated}")
        print(f"  optim_model calls/expansion = {optim_calls_per_expansion:.1f}")
        print(f"  Total optim_model calls   = {tras_total_optim_calls:.1f}")
        print(f"  Total base_model calls    = {tras_total_base_calls}")
        print()
        print("-" * 65)
        print()
        print("Recommended APE Configuration (matched on nodes evaluated):")
        print(f"  num_subsamples            = {num_subsamples}")
        print(f"  num_demos                 = {num_demos}")
        print(f"  num_prompts_per_subsample = {num_prompts_per_subsample}")
        print(f"  total_candidates          = {actual_total}")
        print()
        print("APE Budget:")
        print(f"  optim_model calls         = {ape_optim_calls} (generation)")
        print(f"  base_model eval calls     = {ape_base_eval_calls} (scoring)")
        print()
        print("Note: APE uses significantly fewer optim_model calls than TRAS")
        print("because it generates candidates in one pass rather than iteratively")
        print("computing gradients. The primary budget alignment is on eval calls.")
        print("=" * 65)

    return result


if __name__ == "__main__":
    # Example: compute APE budget for a typical TRAS config
    compute_ape_budget_from_tras(
        iteration_num=12,
        expand_width=3,
        num_new_prompts=1,
        gradient_sampling=2,
        positive_reinforcement_depth=2,
        depth_limit=8,
        train_batch_size=5,
        num_demos=5,
    )
