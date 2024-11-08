
Given source features :math:`f_S` and target features :math:`f_T`, the covariance matrices are given by

    math
$$ C_S = \frac{1}{n_S-1}(f_S^Tf_S-\frac{1}{n_S}(\textbf{1}^Tf_S)^T(\textbf{1}^Tf_S)) $$
math
$$ C_T = \frac{1}{n_T-1}(f_T^Tf_T-\frac{1}{n_T}(\textbf{1}^Tf_T)^T(\textbf{1}^Tf_T)) $$

where :math:`\textbf{1}` denotes a column vector with all elements equal to 1, :math:`n_S, n_T` denotes number of
source and target samples, respectively. We use :math:`d` to denote feature dimension, use
:math:`{\Vert\cdot\Vert}^2_F` to denote the squared matrix `Frobenius norm`. The correlation alignment loss is
given by



math
   $$ l_{CORAL} = \frac{1}{4d^2}\Vert C_S-C_T \Vert^2_F $$

Inputs
    - f_s (tensor): feature representations on source domain, :math:`f^s`
    - f_t (tensor): feature representations on target domain, :math:`f^t`

Shape
    - f_s, f_t: :math:`(N, d)` where d means the dimension of input features, :math:`N=n_S=n_T` is mini-batch size.
    - Outputs: scalar.
