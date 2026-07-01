# Paper reference: OmniMVS32 + Lreg + Lent, Table 3, test split
PAPER_REF = {
    "Urban_Sunny":  {"e1": 13.57, "e3": 4.81, "e5": 3.10, "mae": 0.88, "rmse": 2.56},
    "Urban_Cloudy": {"e1": 13.59, "e3": 4.81, "e5": 3.15, "mae": 0.87, "rmse": 2.53},
    "Urban_Sunset": {"e1": 13.36, "e3": 4.71, "e5": 2.93, "mae": 0.87, "rmse": 2.53},
    "House":        {"e1": 19.89, "e3": 5.89, "e5": 3.99, "mae": 1.30, "rmse": 2.64},
    "Things":       {"e1": 20.70, "e3": 8.18, "e5": 5.49, "mae": 1.37, "rmse": 4.11},
}

_W = {"ds": 16, "split": 7, "frames": 7, "e1": 7, "e3": 7, "e5": 7, "mae": 7, "rmse": 8}
SEP = "─" * (sum(_W.values()) + 2 * len(_W))


def hdr():
    return (
        f"{'Dataset':<{_W['ds']}} {'Split':<{_W['split']}} {'Frames':>{_W['frames']}} "
        f"{'E1(%)':>{_W['e1']}} {'E3(%)':>{_W['e3']}} {'E5(%)':>{_W['e5']}} "
        f"{'MAE(%)':>{_W['mae']}} {'RMSE(%)':>{_W['rmse']}}"
    )


def row(ds_name, split, frames, m):
    if m is None:
        return f"{'  ' + ds_name:<{_W['ds']}} {split:<{_W['split']}} {'—':>{_W['frames']}}  (no data)"
    return (
        f"{'  ' + ds_name:<{_W['ds']}} {split:<{_W['split']}} {frames:>{_W['frames']}} "
        f"{m['e1']:>{_W['e1']}.2f} {m['e3']:>{_W['e3']}.2f} {m['e5']:>{_W['e5']}.2f} "
        f"{m['mae']:>{_W['mae']}.2f} {m['rmse']:>{_W['rmse']}.2f}"
    )


def ref_row(ds_name, ref):
    return (
        f"{'  ' + ds_name:<{_W['ds']}} {'test':<{_W['split']}} {'—':>{_W['frames']}} "
        f"{ref['e1']:>{_W['e1']}.2f} {ref['e3']:>{_W['e3']}.2f} {ref['e5']:>{_W['e5']}.2f} "
        f"{ref['mae']:>{_W['mae']}.2f} {ref['rmse']:>{_W['rmse']}.2f}"
    )
