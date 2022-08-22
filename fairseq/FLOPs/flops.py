def compute_flops(net, input):
    # Set attribute __flops__ for net if not exist
    if not hasattr(net, "__flops__"):
        setattr(net, "__flops__", 0)
        # Generate functions for flops calculation for each module
    # Compute __flops__ for each module recursively
    net.__flops__ = _compute_flops(net, input, FLOPs=[0])
    return net.__flops__


def _compute_flops(_net, input, FLOPs=[0]):
    # Compute __flops__ for each module recursively
    print("Net: ", _net.__class__)
    childrens = list(_net.children())
    if (not childrens) or ("Attention" in str(_net.__class__)):
        # print(_net)
        FLOPs.append(1)
    else:
        for c in childrens:
            _compute_flops(c, input, FLOPs)
    return sum(FLOPs)

    
    
