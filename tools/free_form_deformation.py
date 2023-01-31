from pygem import FFD
import numpy as np
from scipy import special
import cupy as cp

class quick_FFD(FFD):
    @property
    def T(self):
        """
        Return the function that deforms the points within the unit cube.

        :rtype: callable
        """
        def gpu_T_mapping(points):
            cp.cuda.Device(0).use()
            (n_rows, n_cols) = points.shape
            (dim_n_mu, dim_m_mu, dim_t_mu) = self.array_mu_x.shape

            # Initialization. In order to exploit the contiguity in memory the
            # following are transposed
            shift_points = cp.zeros((n_rows, n_cols))

            # time below
            points = cp.asarray(points)
            p1 = 1-points
            bernstein_x = (
                cp.power((p1[:, 0][:, None]), cp.arange(dim_n_mu-1, -1, -1)) *
                cp.power(points[:, 0][:, None], cp.arange(dim_n_mu)) *
                cp.asarray(special.binom(np.array([dim_n_mu-1]*dim_n_mu), np.arange(dim_n_mu)))
            )
            bernstein_y = (
                cp.power((p1[:, 1][:, None]), cp.arange(dim_m_mu-1, -1, -1)) *
                cp.power(points[:, 1][:, None], cp.arange(dim_m_mu)) *
                cp.asarray(special.binom(np.array([dim_m_mu-1]*dim_m_mu), np.arange(dim_m_mu)))
            )
            bernstein_z = (
                cp.power((p1[:, 2][:, None]), cp.arange(dim_t_mu-1, -1, -1)) *
                cp.power(points[:, 2][:, None], cp.arange(dim_t_mu)) *
                cp.asarray(special.binom(np.array([dim_t_mu-1]*dim_t_mu), np.arange(dim_t_mu)))
            )
            bernstein_xyz = bernstein_x[:, :, None, None] * bernstein_y[:, None, :, None] * bernstein_z[:, None, None, :]
            # bernstein_xyz = np.einsum('pi,pj,pk->pijk', bernstein_x, bernstein_y, bernstein_z)
            aux_x = cp.einsum('pijk, ijk', bernstein_xyz, self.array_mu_x)
            aux_y = cp.einsum('pijk, ijk', bernstein_xyz, self.array_mu_y)
            aux_z = cp.einsum('pijk, ijk', bernstein_xyz, self.array_mu_z)
            # array_mu = np.stack([self.array_mu_x, self.array_mu_y, self.array_mu_z], axis=0)
            # aux = np.einsum('pijk, cijk->pc', bernstein_xyz, array_mu)
            # shift_points += aux
            shift_points[:, 0] += aux_x
            shift_points[:, 1] += aux_y
            shift_points[:, 2] += aux_z
            return cp.asnumpy(shift_points + points)

        def T_mapping(points):
            (n_rows, n_cols) = points.shape
            (dim_n_mu, dim_m_mu, dim_t_mu) = self.array_mu_x.shape

            # Initialization. In order to exploit the contiguity in memory the
            # following are transposed
            shift_points = np.zeros((n_rows, n_cols))

            # time below
            import time
            start = time.time()
            p1 = 1-points
            bernstein_x = (
                np.power((p1[:, 0][:, None]), range(dim_n_mu-1, -1, -1)) *
                np.power(points[:, 0][:, None], range(dim_n_mu)) *
                special.binom(np.array([dim_n_mu-1]*dim_n_mu), np.arange(dim_n_mu))
            )
            bernstein_y = (
                np.power((p1[:, 1][:, None]), range(dim_m_mu-1, -1, -1)) *
                np.power(points[:, 1][:, None], range(dim_m_mu)) *
                special.binom(np.array([dim_m_mu-1]*dim_m_mu), np.arange(dim_m_mu))
            )
            bernstein_z = (
                np.power((p1[:, 2][:, None]), range(dim_t_mu-1, -1, -1)) *
                np.power(points[:, 2][:, None], range(dim_t_mu)) *
                special.binom(np.array([dim_t_mu-1]*dim_t_mu), np.arange(dim_t_mu))
            )
            end = time.time()
            bernstein_xyz = bernstein_x[:, :, None, None] * bernstein_y[:, None, :, None] * bernstein_z[:, None, None, :]
            # bernstein_xyz = np.einsum('pi,pj,pk->pijk', bernstein_x, bernstein_y, bernstein_z)
            aux_x = np.einsum('pijk, ijk', bernstein_xyz, self.array_mu_x)
            aux_y = np.einsum('pijk, ijk', bernstein_xyz, self.array_mu_y)
            aux_z = np.einsum('pijk, ijk', bernstein_xyz, self.array_mu_z)
            end1 = time.time()
            print(end-start, end1-end)
            # array_mu = np.stack([self.array_mu_x, self.array_mu_y, self.array_mu_z], axis=0)
            # aux = np.einsum('pijk, cijk->pc', bernstein_xyz, array_mu)
            # shift_points += aux
            shift_points[:, 0] += aux_x
            shift_points[:, 1] += aux_y
            shift_points[:, 2] += aux_z
            return shift_points + points
        
        def gpu_T(points):
            c = np.product(self.array_mu_x.shape)
            p = len(points)
            # c*p <= 16**3*128**3//8
            max_p = 16**3*128**3//8//c
            deformed_points = []
            for i in range(0, p, max_p):
                deformed_points.append(gpu_T_mapping(points[i:i+max_p]))
            cp.get_default_memory_pool().free_all_blocks()
            return np.concatenate(deformed_points)
        return gpu_T

if __name__=='__main__':
    # create a FFD
    l=16
    qffd = quick_FFD([l,l,l])
    ffd  = FFD([l,l,l])
    ffd.array_mu_x = qffd.array_mu_x = np.random.rand(l,l,l)
    ffd.array_mu_y = qffd.array_mu_y = np.random.rand(l,l,l)
    ffd.array_mu_z = qffd.array_mu_z = np.random.rand(l,l,l)

    points = np.random.rand(128**3,3)
    # time qffd and ffd
    import time
    t0 = time.time()
    for i in range(2):
        a = qffd(points)
        print('qffd', time.time()-t0)
    t0 = time.time()
    for i in range(2):
        b = ffd(points)
        print('ffd', time.time()-t0)
    print(np.allclose(a,b), abs(a-b).max())
    # t0 = time.time()
    # for i in range(10):
    #     a[None]
    # print('ffd', time.time()-t0)

