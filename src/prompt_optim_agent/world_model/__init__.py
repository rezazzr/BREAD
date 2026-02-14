from .world_model import WorldModel

WORLD_MODELS = {
    "mcts": WorldModel,
}


def get_world_model(world_model_name):
    if world_model_name not in WORLD_MODELS:
        raise ValueError(
            f"World model '{world_model_name}' is not supported. "
            f"Available: {list(WORLD_MODELS.keys())}"
        )
    return WORLD_MODELS[world_model_name]


__all__ = ["get_world_model", "WorldModel"]
