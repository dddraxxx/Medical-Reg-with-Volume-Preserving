import torch

def sample_power(l, h, k, size=None):
    """
    Sample points from a power law distribution.

    Parameters:
    l (float): lower bound
    h (float): upper bound
    k (float): power 
    size (tuple): number of samples

    Returns:
    tensor: samples
    """
    r = ((h-l)/2)
    # uniform for size
    points = (torch.rand(size)-0.5)*2
    points = torch.abs(points)**k * torch.sign(points) * r
    return points + (l+h)/2

def get_coef(u):
    return torch.stack([((1 - u) ** 3) / 6, (3 * (u ** 3) - 6 * (u ** 2) + 4) / 6,
                     (-3 * (u ** 3) + 3 * (u ** 2) + 3 * u + 1) / 6, (u ** 3) / 6], axis=1)

def pad_3d(mat, pad): return torch.nn.functional.pad(mat, (*[pad, pad], *[pad, pad], *[pad, pad], *[0,0], *[0,0], ), mode='constant')

def free_form_fields(shape, control_fields: torch.Tensor, padding='same'):
    '''Calculate a flow fields based on 3-order B-Spline interpolation of control points.

    Arguments
    --------------
    shape : list of 3 integers, flow field shape `(x, y, z)`
    control_fields : 5d tensor with 3 channels `(batch_size, 3, n, m, t)`

    Output
    --------------
    5d tensor with 3 channels `(batch_size, 3, x, y, z)`
    '''
    interpolate_range = 4
    _, _, n, m, t = control_fields.shape
    if padding=='same':
        control_fields = pad_3d(control_fields, 1)
    elif padding=='valid':
        n -= 2
        m -= 2
        t -= 2
    control_fields = control_fields.permute(2, 3, 4, 0, 1).reshape(n+2, m+2, t+2, -1)
    shape_cf = control_fields.shape

    assert shape[0] % (n-1) == 0 and shape[1] % (m-1) == 0 and shape[2] % (t-1) == 0, 'Shape must be divisible by control points'
    s_x = shape[0] // (n-1)
    u_x = (torch.arange(0, s_x, dtype=torch.float32)+0.5) / s_x
    u_x = u_x.to(control_fields.device)
    coef_x = get_coef(u_x)
    flow = torch.concat([coef_x @ control_fields[i: i+interpolate_range].reshape(interpolate_range, -1) 
        for i in range(n-1)], axis=0)
    
    s_y = shape[1] // (m-1)
    u_y = (torch.arange(0, s_y, dtype=torch.float32)+0.5) / s_y
    u_y = u_y.to(control_fields.device)
    coef_y = get_coef(u_y)
    flow = flow.t().reshape(shape_cf[1], -1)
    flow = torch.concat([coef_y @ flow[i: i+interpolate_range].reshape(interpolate_range, -1)
        for i in range(m-1)], axis=0)
    
    s_z = shape[2] // (t-1)
    u_z = (torch.arange(0, s_z, dtype=torch.float32)+0.5) / s_z
    u_z = u_z.to(control_fields.device)
    coef_z = get_coef(u_z)
    flow = flow.t().reshape(shape_cf[2], -1)
    flow = torch.concat([coef_z @ flow[i: i+interpolate_range].reshape(interpolate_range, -1)
        for i in range(t-1)], axis=0)
    flow = flow.reshape(shape[2], -1, 3, shape[1], shape[0]).permute(1, 2, 4, 3, 0)
    return flow

if __name__ == '__main__':
    import numpy as np
    # seed np
    np.random.seed(0)
    # get random int array
    arr = np.random.randint(0, 10, (1,3,5,5,5))
    
    shape = [128,128,128]
    control_fields = torch.tensor(arr, dtype=torch.float32)
    flow = free_form_fields(shape, control_fields)
    print(flow.shape, flow.permute(0,2,3,4,1))
    print(control_fields)



    