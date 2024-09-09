import torch
import warnings

def convol_3d(cloud, K):
        kx = torch.einsum("ij,rjlk->rilk", K, cloud)
        kxy = torch.einsum("ij,rkjl->rkil", K, kx)
        kxyz = torch.einsum("ij,rlkj->rlki", K, kxy)
        return kxyz


def barycenter_debiased_1d(P, M, reg, maxiter=5000, tol=1e-5, weights=None):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    P = torch.as_tensor(P).clone().detach().to(device)
    M = torch.as_tensor(M).clone().detach().to(device)
    
    dim, n_hists = P.shape
    
    
    K = torch.exp(-M/reg)
    Ka = torch.ones_like(P, device=P.device)
    Kb = torch.ones_like(P, device=P.device)
    
    b = torch.ones_like(P, device=P.device)

    d, bar = torch.ones((2, dim), dtype=P.dtype, device=P.device)
    
   
    if weights is None:
        weights = torch.ones(n_hists, dtype=P.dtype, device=P.device) / n_hists
    else:
        assert (len(weights) == P.shape[1])
        
    for ii in range(maxiter):
        bar_old = bar.clone()
        a = P / Kb
        Ka = K.t().mm(a)
        bar = d * torch.prod((Ka) ** weights[None, :], dim=1)
        
        b = bar[:,None] / Ka
        Kb = K.mm(b)

        
        d = (d * bar / K.mv(d)) ** 0.5
        
      
        if abs(bar - bar_old).max() < tol and ii > 10:
            break
    if ii == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))
    return bar


def IBP_1d(P, M, reg, maxiter=5000, tol=1e-5, weights=None):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    P = torch.as_tensor(P).clone().detach().to(device)
    M = torch.as_tensor(M).clone().detach().to(device)
    
    dim, n_hists = P.shape
    
    
    K = torch.exp(-M/reg)
    Ka = torch.ones_like(P, device=P.device)
    Kb = torch.ones_like(P, device=P.device)
    
    b = torch.ones_like(P, device=P.device)

    bar = torch.ones(dim, dtype=P.dtype, device=P.device)
    
   
    if weights is None:
        weights = torch.ones(n_hists, dtype=P.dtype, device=P.device) / n_hists
    else:
        assert (len(weights) == P.shape[1])
        
    for ii in range(maxiter):
        bar_old = bar.clone()
        a = P / Kb
        Ka = K.t().mm(a)
        bar = torch.prod((Ka) ** weights[None, :], dim=1)
        
        b = bar[:,None] / Ka
        Kb = K.mm(b)
        
      
        if abs(bar - bar_old).max() < tol and ii > 10:
            break
    if ii == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))
    return bar



def _barycenter_inner_1d(P, K, qold=None, bold=None, maxiter=1000,
                         tol=1e-4, weights=None):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    dim, n_hists = P.shape
    if bold is None:
        bold = torch.ones_like(P)
    b = bold.clone()
    if qold is None:
        qold = torch.ones(dim) / dim
    Kb = K.mm(b)
    err = 10
    if weights is None:
        weights = torch.ones(n_hists) / n_hists
    q = qold.clone()
    for ii in range(maxiter):
        qold_inner = q.clone()
        a = P / Kb
        Ka = K.t().mm(a)
        q = qold * torch.prod((Ka) ** weights[None, :], dim=1)
        Q = q[:, None]
        b = Q / Ka
        Kb = K.mm(b)
        err = abs(q - qold_inner).max()
        if err < tol and ii > 10:
            break
    if ii == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))

    return q, b


def barycenter_product_1d(P, M, reg, maxiter=500, tol=1e-5, weights=None):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    P = torch.as_tensor(P).clone().detach().to(device)
    M = torch.as_tensor(M).clone().detach().to(device)

    dim, n_hists = P.shape
   

    q = torch.ones(dim) / dim
    b = torch.ones_like(P)

    K = torch.exp(-M/reg)
    for ii in range(maxiter):
        qold = q.clone()
        q, b = _barycenter_inner_1d(P, K, qold=q, bold=b)
        err = abs(q - qold).max()
        if err < tol:
            break
    return q


def barycenter_debiased_2d(P, M, reg, maxiter=5000, tol=1e-5, weights=None):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    def convol_imgs(imgs, K):
        kx = torch.einsum("...ij,kjl->kil", K, imgs)
        kxy = torch.einsum("...ij,klj->kli", K, kx)
        return kxy
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    P = torch.as_tensor(P).clone().detach().to(device)
    M = torch.as_tensor(M).clone().detach().to(device)
    
    n_hists, width, _ = P.shape
    
    K = torch.exp(-M / reg).to(device)
    Ka = torch.ones_like(P, device=device)    
    
    b = torch.ones_like(P, device=device)
    Kb = convol_imgs(b, K)

    bar = torch.ones((width, width), dtype=P.dtype, device=device)
    d = torch.ones((width, width), dtype=P.dtype, device=device)
   
    if weights is None:
        weights = torch.ones(n_hists, dtype=P.dtype, device=device) / n_hists
    else:
        weights = torch.as_tensor(weights, dtype=P.dtype, device=device)
        assert len(weights) == P.shape[0]
        
    for ii in range(maxiter):
        bar_old = bar.clone()
        a = P / Kb
        Ka = convol_imgs(a, K.t())
        bar = d * torch.prod((Ka) ** weights[:, None, None], dim=0)
        
        b = bar[None, :, :] / Ka
        Kb = convol_imgs(b, K)
        
        for kk in range(10):
            Kd = K.t().mm(K.mm(d).t()).t()
            d = (d * bar / Kd) ** 0.5
        
        if abs(bar - bar_old).max() < tol and ii > 10:
            break
    
    if ii == maxiter - 1:
        warnings.warn(f"*** Maxiter reached ! err = {abs(bar - bar_old).max()} ***")
    
    return bar


def IBP_2d(P, M, reg, maxiter=5000, tol=1e-5, weights=None):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    def convol_imgs(imgs, K):
        kx = torch.einsum("...ij,kjl->kil", K, imgs)
        kxy = torch.einsum("...ij,klj->kli", K, kx)
        return kxy
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    P = torch.as_tensor(P).clone().detach().to(device)
    M = torch.as_tensor(M).clone().detach().to(device)
    
    n_hists, width, _ = P.shape
    
    K = torch.exp(-M / reg).to(device)

    
    b = torch.ones_like(P, requires_grad=False)
    bar = torch.ones((width, width), dtype=P.dtype, device=P.device)
    Kb = convol_imgs(b, K)
    err = 1
    if weights is None:
        weights = torch.ones(n_hists, dtype=P.dtype, device=P.device) / n_hists
    for ii in range(maxiter):
        bar_old = bar.clone()
        a = P / Kb
        Ka = convol_imgs(a, K.t())
        bar = torch.prod((b * Ka) ** weights[:, None, None], dim=0)
        b = bar[None, :, :] / Ka
        Kb = convol_imgs(b, K)
        
        err = abs(bar - bar_old).max()

        if err < tol and ii > 10:
            break
    if ii == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))
    return bar



def _barycenter_inner_2d(P, M, reg, bar_old=None, bold=None, maxiter=1000,
                         tol=1e-4, weights=None):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    def convol_imgs(imgs, K):
        kx = torch.einsum("...ij,kjl->kil", K, imgs)
        kxy = torch.einsum("...ij,klj->kli", K, kx)
        return kxy
    
    n_hists, width, _ = P.shape
    if bold is None:
        bold = torch.ones_like(P, requires_grad=False)
    K = torch.exp(-M/reg)    
    
    b = bold.clone()
    Kb = convol_imgs(b, K)
    if weights is None:
        weights = torch.ones(n_hists, dtype=P.dtype, device=P.device) / n_hists
    if bar_old is None:
        bar_old = torch.ones_like(P[0]) / (width ** 2)
    bar = bar_old.clone()
    for ii in range(maxiter):
        bar_local = bar.clone()
        a = P / Kb
        Ka = convol_imgs(a, K.t())
        bar = bar_old * torch.prod(Ka ** weights[:, None, None], dim=0)
        b = bar[None, :, :] / Ka
        Kb = convol_imgs(b, K)
        err = abs(bar - bar_local).max()
        if err < tol and ii > 10:
            break

    if ii == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))

    return bar, b

def barycenter_product_2d(P, M, reg, maxiter=500, tol=1e-5, weights=None):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    P = torch.as_tensor(P).clone().detach().to(device)
    M = torch.as_tensor(M).clone().detach().to(device)
    
    n_hists, width, _ = P.shape
    bar = torch.ones((width, width), device=P.device, dtype=P.dtype)
    b = torch.ones_like(P)
    for ii in range(maxiter):
        bar_old = bar.clone()
        bar, b = _barycenter_inner_2d(P, M, reg, bar_old=bar, bold=b)
        err = abs(bar - bar_old).max()
        if err < tol:
            break
    return bar


def barycenter_debiased_3d(P, M, reg, maxiter=5000, tol=1e-5, weights=None):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    P = torch.as_tensor(P).clone().detach().to(device)
    M = torch.as_tensor(M).clone().detach().to(device)
    
    n_hists, width, _, _ = P.shape
    
    K = torch.exp(-M / reg).to(device)
    Ka = torch.ones_like(P, device=device)    
    
    b = torch.ones_like(P, device=device)
    Kb = convol_3d(b, K)

    bar = torch.ones((width, width,width), dtype=P.dtype, device=device)
    d = torch.ones((width, width,width), dtype=P.dtype, device=device)
   
    if weights is None:
        weights = torch.ones(n_hists, dtype=P.dtype, device=device) / n_hists
    else:
        weights = torch.as_tensor(weights, dtype=P.dtype, device=device)
        assert len(weights) == P.shape[0]
        
    for ii in range(maxiter):
        bar_old = bar.clone()
        a = P / Kb
        Ka = convol_3d(a, K.t())
        bar = d * torch.prod((Ka) ** weights[:, None, None, None], dim=0)
        
        b = bar[None, :, :, :] / Ka
        Kb = convol_3d(b, K)
        
        for kk in range(10):
            Kd = convol_3d(d[None, :], K).squeeze()
            d = (d * bar / Kd) ** 0.5
        
        if abs(bar - bar_old).max() < tol and ii > 10:
            break
    
    if ii == maxiter - 1:
        warnings.warn(f"*** Maxiter reached ! err = {abs(bar - bar_old).max()} ***")
    
    return bar


def IBP_3d(P, M, reg, maxiter=5000, tol=1e-5, weights=None):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    P = torch.as_tensor(P).clone().detach().to(device)
    M = torch.as_tensor(M).clone().detach().to(device)
    
    n_hists, width, _, _ = P.shape
    
    K = torch.exp(-M / reg).to(device)

    
    b = torch.ones_like(P, requires_grad=False)
    bar = torch.ones((width, width,width), dtype=P.dtype, device=P.device)
    Kb = convol_3d(b, K)
    err = 1
    if weights is None:
        weights = torch.ones(n_hists, dtype=P.dtype, device=P.device) / n_hists
    for ii in range(maxiter):
        bar_old = bar.clone()
        a = P / Kb
        Ka = convol_3d(a, K.t())
        bar = torch.prod((b * Ka) ** weights[:, None, None, None], dim=0)
        b = bar[None, :, :, :] / Ka
        Kb = convol_3d(b, K)
        
        err = abs(bar - bar_old).max()

        if err < tol and ii > 10:
            break
    if ii == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))
    return bar


def barycenter_3d(P, K, Kb=None, c=None, maxiter=1000, tol=1e-7,
                  debiased=False, weights=None, return_log=False):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    n_hists, width, _, _ = P.shape
    b = torch.ones_like(P, requires_grad=False)
    q = torch.ones((width, width, width), device=P.device, dtype=P.dtype)
    if Kb is None:
        Kb = convol_3d(b, K)
    if c is None:
        c = q.clone()
    log = {'err': [], 'a': [], 'b': [], 'q': []}
    err = 10
    if weights is None:
        weights = torch.ones(n_hists, device=P.device, dtype=P.dtype) / n_hists
    for ii in range(maxiter):
        if torch.isnan(q).any():
            break
        qold = q.clone()
        a = P / Kb
        Ka = convol_3d(a, K.t())
        q = c * torch.prod((Ka) ** weights[:, None, None, None], dim=0)
        if debiased:
            Kc = convol_3d(c[None, :], K).squeeze()
            c = (c * q / Kc) ** 0.5
        Q = q[None, :]
        b = Q / Ka
        Kb = convol_3d(b, K)
        err = abs(q - qold).max()

        if err < tol and ii > 10:
            break
    print("Barycenter 3d | err = ", err)
    if return_log:
        log["err"].append(err)
        log["a"] = a
        log["q"] = q
        log["b"] = b

    if ii == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))

    if return_log:
        return q, log
    return q