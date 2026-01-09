

def extend_and_repeat(state, dim: int, repeat: int):
    if isinstance(state, tuple):
        return tuple(
            s.unsqueeze(dim).repeat_interleave(repeat, dim=dim)
            for s in state
        )
    else:
        return state.unsqueeze(dim).repeat_interleave(repeat, dim=dim)