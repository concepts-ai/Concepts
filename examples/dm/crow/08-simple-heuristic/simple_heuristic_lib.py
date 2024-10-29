import jacinle
import torch
import concepts.dm.crow as crow

walls = {(1, 1)}


@crow.config_function_implementation
def l1_distance(loc1: torch.Tensor, loc2: torch.Tensor) -> float:
    return torch.sum(torch.abs(loc1 - loc2))


@crow.config_function_implementation
def wall_at(loc: torch.Tensor) -> bool:
    x = tuple(loc.tolist())

    if x[0] < 0 or x[0] > 3 or x[1] < 0 or x[1] > 3:
        return True

    if x in walls:
        return True

    return False

