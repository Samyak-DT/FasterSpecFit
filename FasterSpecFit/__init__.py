from .emlines_objective import (
    objective            as EMLine_objective,
    jacobian             as EMLine_jacobian,
    find_peak_amplitudes as EMLine_find_peak_amplitudes,
    prepare_bins         as EMLine_prepare_bins
)

from .params_mapping import ParamsMapping as EMLine_ParamsMapping

from .sparse_rep import ResMatrix
