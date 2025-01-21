import numpy as np
import torch
import torch.nn as nn
from scipy import optimize
import time

def Tcons(arr, use_cuda):
    """
    Converts a numpy array to a torch tensor and moves it to GPU if specified.

    Args:
        arr (numpy.ndarray): Input array.
        use_cuda (bool): Whether to move the tensor to GPU.

    Returns:
        torch.Tensor: Converted tensor.
    """
    temp = torch.Tensor(arr)
    if use_cuda:
        temp = temp.to('cuda')
    return temp

class Taylor_inverse(nn.Module):
    """
    Computes the inverse of a matrix using Taylor series approximation.

    Args:
        dim (int): Dimension of the square matrix.
        iters (int): Number of Taylor series terms to compute.
    """
    def __init__(self, dim, iters=1):
        super(Taylor_inverse, self).__init__()
        self.dim = dim
        self.iters = iters

        # Initialize buffers to store intermediate computations
        self.register_buffer('M_', torch.zeros((dim, dim)))
        self.register_buffer('Mi', torch.eye(dim))

    def forward(self, M):
        """
        Perform Taylor series approximation of the matrix inverse.

        Args:
            M (torch.Tensor): Input square matrix.

        Returns:
            torch.Tensor: Approximated inverse of the input matrix.
        """
        M = M.to(self.Mi.device)
        self.M_.zero_()
        self.Mi.fill_diagonal_(1)  # Reset Mi to the identity matrix

        # Iterative Taylor series computation
        for i in range(self.iters):
            self.Mi = torch.matmul(self.Mi, M)
            self.M_ = self.M_ + self.Mi

        invM = torch.eye(self.dim, device=M.device) + self.M_
        return invM


def NMFupdate_KL(V, W, H):
    """
    One iteration of NMF update using Kullback-Leibler divergence.

    Args:
        V (torch.Tensor): Input data matrix.
        W (torch.Tensor): Basis matrix.
        H (torch.Tensor): Coefficient matrix.

    Returns:
        tuple: Updated (W, H).
    """
    # Update H
    Hout = H * (W.t() @ (V / (W @ H + 1e-10))) / (W.t() @ torch.ones_like(V) + 1e-10)
    # Update W
    Wout = W * ((V / (W @ Hout + 1e-10)) @ Hout.t()) / (torch.ones_like(V) @ Hout.t() + 1e-10)
    return Wout, Hout


def NMFiter_KL(Xinit, nIter, Winit, Hinit, retain_intermediates=False):
    """
    Performs multiple iterations of NMF updates using Kullback-Leibler divergence.

    Args:
        Xinit (torch.Tensor): Input data matrix.
        nIter (int): Number of iterations.
        Winit (torch.Tensor): Initial basis matrix.
        Hinit (torch.Tensor): Initial coefficient matrix.
        retain_intermediates (bool): Whether to retain intermediate matrices.

    Returns:
        tuple: Final (W, H) matrices or lists of intermediates if `retain_intermediates` is True.
    """
    Wcur = [Winit]
    Hcur = [Hinit]
    for i in range(nIter):
        Wout, Hout = NMFupdate_KL(Xinit, Wcur[-1], Hcur[-1])
        if retain_intermediates:
            Wcur.append(Wout)
            Hcur.append(Hout)
        else:
            Wcur[0] = Wout
            Hcur[0] = Hout
    return (Wcur, Hcur) if retain_intermediates else (Wcur[0], Hcur[0])



def loss_w(Worig, Wfin):
    """
    Compute the error between two basis matrices (Worig and Wfin).

    Args:
        Worig (torch.Tensor): Original basis matrix.
        Wfin (torch.Tensor): Final basis matrix.

    Returns:
        torch.Tensor: Normalized error between the matrices.
    """
    if Worig.shape != Wfin.shape:
        raise ValueError("W matrices are not of the same shape")

    # Normalize features of W matrices
    WorigN = Worig / torch.norm(Worig, dim=0, keepdim=True)
    WfinN = Wfin / torch.norm(Wfin, dim=0, keepdim=True)

    # Compute pairwise errors
    errMat = torch.norm(WorigN.unsqueeze(2) - WfinN.unsqueeze(1), dim=0).T**2

    # Find the optimal matching
    order = optimize.linear_sum_assignment(errMat.cpu().numpy())[1]
    WfinNS = WfinN[:, order]

    # Compute the overall error
    error = torch.norm(WorigN - WfinNS) / torch.norm(WorigN)
    return error

    
def loss_wh(Worig, Wfin, Horig, Hfin):
    """
    Computes the normalized error between the original and final factor matrices (W, H)
    for given PyTorch tensors.

    Args:
        Worig (torch.Tensor): Original basis matrix (W).
        Wfin (torch.Tensor): Final basis matrix (W) after optimization.
        Horig (torch.Tensor): Original coefficient matrix (H).
        Hfin (torch.Tensor): Final coefficient matrix (H) after optimization.

    Returns:
        torch.Tensor: Normalized error between the original and final factor matrices.
    """
    # Ensure the matrices Worig and Wfin have the same shape
    if Worig.shape != Wfin.shape:
        raise ValueError("W matrices are not of the same shape")

    # Ensure the matrices Horig and Hfin have the same shape
    if Horig.shape != Hfin.shape:
        raise ValueError("H matrices are not of the same shape")

    # Compute balancing factors for W and H based on their norms
    tNormOrig = torch.sqrt(torch.norm(Worig, dim=0) * torch.norm(Horig, dim=1))
    tNormFin = torch.sqrt(torch.norm(Wfin, dim=0) * torch.norm(Hfin, dim=1))

    # Balance the columns of W and the rows of H using their norms
    WorigB = Worig * (tNormOrig / torch.norm(Worig, dim=0).unsqueeze(0))
    HorigB = (Horig.T * (tNormOrig / torch.norm(Horig, dim=1).T.unsqueeze(0))).T
    WfinB = Wfin * (tNormFin / torch.norm(Wfin, dim=0).unsqueeze(0))
    HfinB = (Hfin.T * (tNormFin / torch.norm(Hfin, dim=1).T.unsqueeze(0))).T

    # Concatenate W and H (balanced) for original and final matrices
    Corig = torch.cat((WorigB, HorigB.T), 0)
    Cfin = torch.cat((WfinB, HfinB.T), 0)

    # Compute a pairwise error matrix between Corig and Cfin
    # The error is calculated as the norm of the difference between feature vectors
    errMat = torch.norm(Corig.unsqueeze(1) - Cfin.unsqueeze(2), dim=0)

    # Square the error matrix to compute total error contributions
    errMat = errMat.T**2

    # Use the Hungarian algorithm (linear_sum_assignment) to find the optimal matching
    # that minimizes the sum of errors between features
    order = optimize.linear_sum_assignment(errMat.cpu().detach().numpy())[1]

    # Reorder the features of Cfin to align with Corig for minimal error
    CfinS = Cfin[:, order]

    # Compute the normalized error between the original and final concatenated matrices
    error = torch.norm(Corig - CfinS) / torch.norm(Corig)

    return error


def loss_rec(V,W,H):
    error = torch.sum(torch.square(W@H - V)) 
    return error

# Gradient-based attacks
class Gradient_based_attack(nn.Module):
    def __init__(self, X, 
                 nmf_rank = 5, 
                 base_nmf_iters = 1000,  
                 use_cuda = True,
                 # average = 1,
                 # pgd = True,
                 implicit_func = True,
                 implicit_w = False,
                 taylor = 100,
                 norm = "Linf", 
                 rec_loss = False,
                 no_batch = 1,
                 verbose = False):
        
        super(Gradient_based_attack, self).__init__()  # Initialize the superclass

        self.X = X
        self.scale = torch.norm(X)
        # self.ave = average
        self.rank = nmf_rank
        self.Wrow = X.shape[0]
        self.Wcol = nmf_rank
        self.Hrow = nmf_rank
        self.Hcol = X.shape[1]
        self.Wdim = self.Wrow*self.Wcol
        self.Hdim = self.Hrow*self.Hcol
        self.Xdim = self.Wrow*self.Hcol
        self.nmf_iters = base_nmf_iters
        self.use_cuda = use_cuda
        self.norm = norm
        self.taylor = taylor
        self.implicit_func = implicit_func
        self.Wstays = []
        self.Hstays = []
        self.rec_loss = rec_loss
        self.implicit_w = implicit_w
        self.no_batch = no_batch

        if self.implicit_func:
            grad_type = "Implicit function"
        else:
            grad_type = "Back-propagate"

        # self.pgd = pgd
        self.grad_type = grad_type

        self.save_stationary()

        if verbose:
            # print("Gradient type: ", self.grad_type)
            # print("Attack norm: ", self.norm)
            print("Xdim: ", self.Xdim)
            print("Wdim: ", self.Wdim)
            print("Hdim: ", self.Hdim)
            print("NMF iteration: ", self.nmf_iters)
            print("NMF rank: ", self.rank)
            # if self.pgd:
            #     print("PGD attack")
            # else:
            #     print("One-step attack")
            print("Gradient type: ", grad_type)
            print("NMF stationary (W,H) error: ", self.origErrors)


    def save_stationary(self):
        Wstay, Hstay, (error_w, error_h) = self.get_stationary(self.X)
        self.Worig = Wstay
        self.Horig = Hstay
        self.origErrors = (error_w, error_h)
    
    def get_stationary(self, X):
        
        Winit = np.random.rand(self.Wrow,self.rank)
        Hinit = np.random.rand(self.rank,self.Hcol)
        Winit = torch.Tensor(Winit)
        Hinit = torch.Tensor(Hinit)
    
        WinitT = Tcons(Winit,self.use_cuda).float()
        HinitT = Tcons(Hinit,self.use_cuda).float()

        Wstay,Hstay = NMFiter_KL(X,self.nmf_iters,WinitT,HinitT)

        # Check stationary
        Wstay_,Hstay_ = NMFiter_KL(X,10,Wstay,Hstay)

        error_w = torch.norm(Wstay_ - Wstay)/torch.norm(Wstay)
        error_h = torch.norm(Hstay_ - Hstay)/torch.norm(Hstay)
            
        return Wstay, Hstay, (error_w, error_h)

    def backpropagate(self, X, return_stay_error = False):
        XpertT = X.clone()
        # XpertT = Tcons(XpertT, self.use_cuda)
        XpertT.requires_grad_()
        Wstay,Hstay, stationary_errors = self.get_stationary(XpertT)

        if self.rec_loss:
            wErr=loss_rec(self.X,Wstay,Hstay)
        else:
            wErr=loss_wh(self.Worig, Wstay, self.Horig, Hstay)
        wErr.backward()

        grad = XpertT.grad
        
        if return_stay_error:
            return grad.reshape((self.Wrow,self.Hcol)), stationary_errors
        else:
            return grad.reshape((self.Wrow,self.Hcol))

    def forward(self, X, return_stay_error = False):
        # Implementation of the implicit function gradients' computation
        
        grad = torch.zeros(self.Xdim)
        grad = Tcons(grad, self.use_cuda)
        with torch.no_grad():
            Wstay,Hstay, stationary_errors = self.get_stationary(X)

        # Base Jacobian
        jacs = torch.autograd.functional.jacobian(NMFupdate_KL,(X,Wstay,Hstay))

        # dydx base
        J = torch.zeros(self.Wdim+self.Hdim,self.Wdim+self.Hdim)
        J = Tcons(J, self.use_cuda)
        J[0:self.Wdim,0:self.Wdim] = jacs[0][1].reshape(self.Wdim,self.Wdim)
        J[0:self.Wdim,self.Wdim:self.Wdim+self.Hdim] = jacs[0][2].reshape(self.Wdim,self.Hdim)
        J[self.Wdim:self.Wdim+self.Hdim,0:self.Wdim] = jacs[1][1].reshape(self.Hdim,self.Wdim)
        J[self.Wdim:self.Wdim+self.Hdim,self.Wdim:self.Wdim+self.Hdim] = jacs[1][2].reshape(self.Hdim,self.Hdim)


        inverter = Taylor_inverse(J.shape[0], iters = self.taylor)
        if self.use_cuda:
            inverter = inverter.to('cuda')
        with torch.no_grad():
            invJ = inverter(J)

        
        # Check Nan
        taylor_order = self.taylor
        while invJ.isnan().any():
            print("Warning: Taylor inverse encounters NaN. Try smaller Taylor's order")
            taylor_order = int(taylor_order*0.9 + 1)
            print("Taylor's order: ", taylor_order)
            inverter = Taylor_inverse(J.shape[0], iters = taylor_order)
            if self.use_cuda:
                inverter = inverter.to('cuda')
            with torch.no_grad():
                invJ = inverter(J)

        with torch.no_grad():
            pfpx = torch.zeros(self.Wdim+self.Hdim,self.Xdim)
            pfpx = Tcons(pfpx, self.use_cuda)
            pfpx[0:self.Wdim,:] = jacs[0][0].reshape(self.Wdim,self.Xdim)
            pfpx[self.Wdim:self.Wdim+self.Hdim,:] = jacs[1][0].reshape(self.Hdim,self.Xdim)
            # print(invJ.shape, pfpx.shape)
            # dydx = torch.mm(invJ, pfpx)
            # dydx = torch.matmul(invJ, pfpx)
            invJ = invJ.cpu()
            pfpx = pfpx.cpu()
            dydx = invJ@pfpx
            # dydx = invJ@pfpx
            # dydx = torch.bmm(invJ.unsqueeze(0).expand_as(pfpx), pfpx)
            dydx = dydx.reshape((self.Wdim + self.Hdim,self.Xdim))

        if self.rec_loss:
            jac_loss = torch.autograd.functional.jacobian(loss_rec,(self.X,Wstay,Hstay))
            grad_dldw = jac_loss[1].reshape(self.Wdim)
            grad_dldh = jac_loss[2].reshape(self.Hdim)
        else:
            jac_loss = torch.autograd.functional.jacobian(loss_wh,(self.Worig, Wstay, self.Horig, Hstay))
            grad_dldw = jac_loss[1].reshape(self.Wdim)
            grad_dldh = jac_loss[3].reshape(self.Hdim)
        
        # grad_dldy = torch.cat((grad_dldw,grad_dldh))
        grad_dldy = torch.cat((grad_dldw,grad_dldh)).cpu()
        grad = grad_dldy@dydx
        grad = Tcons(grad, self.use_cuda)

        if return_stay_error:
            return grad.reshape((self.Wrow,self.Hcol)), stationary_errors
        else:
            return grad.reshape((self.Wrow,self.Hcol))
        
    def forward_batch(self, X, return_stay_error = False):
        # Implementation of the implicit function gradients' computation with batching
        
        grad = torch.zeros(self.Xdim)
        grad = Tcons(grad, self.use_cuda)
        with torch.no_grad():
            Wstay,Hstay, stationary_errors = self.get_stationary(X)
            
        no_batch = self.no_batch
        X_batches = torch.tensor_split(X, no_batch, dim=0)
        W_batches = torch.tensor_split(Wstay, no_batch, dim=0)
        Worig_batches = torch.tensor_split(self.Worig, no_batch, dim=0)
        grads = []

        for i in range(no_batch):
            jacs = torch.autograd.functional.jacobian(NMFupdate_KL,(X_batches[i],W_batches[i],Hstay))

            # dydx base
            Wdim = int(self.Wdim/no_batch)
            J = torch.zeros(Wdim+self.Hdim,Wdim+self.Hdim)
            J = Tcons(J, self.use_cuda)
            J[0:Wdim,0:Wdim] = jacs[0][1].reshape(Wdim,Wdim)
            J[0:Wdim,Wdim:Wdim+self.Hdim] = jacs[0][2].reshape(Wdim,self.Hdim)
            J[Wdim:Wdim+self.Hdim,0:Wdim] = jacs[1][1].reshape(self.Hdim,Wdim)
            J[Wdim:Wdim+self.Hdim,Wdim:Wdim+self.Hdim] = jacs[1][2].reshape(self.Hdim,self.Hdim)

            inverter = Taylor_inverse(J.shape[0], iters = self.taylor)
            if self.use_cuda:
                inverter = inverter.to('cuda')
            with torch.no_grad():
                invJ = inverter(J)
                
            # Check Nan
            taylor_order = self.taylor
            while invJ.isnan().any():
                print("Warning: Taylor inverse encounters NaN. Try smaller Taylor's order")
                taylor_order = int(taylor_order*0.8 + 1)
                print("Taylor's order: ", taylor_order)
                inverter = Taylor_inverse(J.shape[0], iters = taylor_order)
                if self.use_cuda:
                    inverter = inverter.to('cuda')
                with torch.no_grad():
                    invJ = inverter(J)
                

            Xdim = int(self.Xdim/no_batch)
            with torch.no_grad():
                pfpx = torch.zeros(Wdim+self.Hdim,Xdim)
                # pfpx = Tcons(pfpx, self.use_cuda)
                pfpx[0:Wdim,:] = jacs[0][0].reshape(Wdim,Xdim)
                pfpx[Wdim:Wdim+self.Hdim,:] = jacs[1][0].reshape(self.Hdim,Xdim)
                invJ = invJ.cpu()
                pfpx = pfpx.cpu()
                dydx = invJ@pfpx
                dydx = dydx.reshape((Wdim + self.Hdim,Xdim))

                

            if self.rec_loss:
                jac_loss = torch.autograd.functional.jacobian(loss_rec,(X_batches[i],W_batches[i],Hstay))
                grad_dldw = jac_loss[1].reshape(Wdim)
                grad_dldh = jac_loss[2].reshape(self.Hdim)
            else:
                jac_loss = torch.autograd.functional.jacobian(loss_wh,(Worig_batches[i], W_batches[i], self.Horig, Hstay))
                grad_dldw = jac_loss[1].reshape(Wdim)
                grad_dldh = jac_loss[3].reshape(self.Hdim)

                

            grad_dldy = torch.cat((grad_dldw,grad_dldh)).cpu()
            grad = grad_dldy@dydx

    

            grad = Tcons(grad, self.use_cuda)
            grads.append(grad)

        grad = torch.cat(grads)

        if return_stay_error:
            return grad.reshape((self.Wrow,self.Hcol)), stationary_errors
        else:
            return grad.reshape((self.Wrow,self.Hcol)) 
        
    def pgd_attack(self, eps=0.001, alpha=1, iters=10, record = False, average_grad = 1):
        self.records = []
        xmax = torch.max(self.X)
        X_ = self.X.clone()
        for i in range(iters):
            grad = torch.zeros((self.Wrow,self.Hcol))
            grad = Tcons(grad, self.use_cuda)
            for _ in range(average_grad):
                if self.implicit_func:
                    if self.no_batch > 1:
                        start_time = time.time()
                        # print("Use batching: ", self.no_batch)
                        grad = grad + self.forward_batch(X_)
                        # print("Duration for 1 grad: ", time.time() - start_time)
                    else:
                        grad = grad + self.forward(X_)
                else:
                    grad = grad + self.backpropagate(X_)
            grad = grad/average_grad
            grad = grad.sign()
            X_ = X_ + alpha*grad

            if self.norm == "Linf":
                eta = torch.clamp(X_ - self.X, min=-eps, max=eps)
            else:
                eta = (X_ - self.X)/torch.norm(X_ - self.X) * eps * self.scale

            X_ = torch.clamp(self.X + eta, min=0, max=xmax)
            if record == True:
                self.records.append(X_)
        return X_
    
    def step_attack(self, eps=0.001, record = False, average_grad = 20):
        self.records = []
        xmax = torch.max(self.X)
        X_ = self.X.clone()

        grad = torch.zeros((self.Wrow,self.Hcol))
        grad = Tcons(grad, self.use_cuda)
        for _ in range(average_grad):
            if self.implicit_func:
                grad = grad + self.forward(X_)
            else:
                grad = grad + self.backpropagate(X_)
        grad = grad/average_grad
        X_ = X_ + grad

        if self.norm == "Linf":
            eta = torch.clamp(X_ - self.X, min=-eps, max=eps)
        else:
            eta = (X_ - self.X)/torch.norm(X_ - self.X) * eps * self.scale
                
        X_ = torch.clamp(self.X + eta, min=0, max=xmax)
        if record == True:
            self.records.append(X_)

        return X_


