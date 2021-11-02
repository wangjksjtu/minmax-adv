import torch
import torch.nn as nn

from ..attack import Attack


def project_simplex(v, z=1.0, axis=-1):
    """
    Implements the algorithm in Figure 1 of
    John Duchi, Shai Shalev-Shwartz, Yoram Singer, Tushar Chandra,
    "Efficient Projections onto the l1-Ball for Learning in High Dimensions", ICML 2008.
    https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    This algorithm project vectors v onto the simplex w >= 0, \sum w_i = z.
    :param v: A torch tensor, will be interpreted as a collection of vectors.
    :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
    :param axis: Indicates the axis of v, which defines the vectors to be projected.
    :return: w: result of the projection
    """

    def _project_simplex_2d(v, z):
        """
        Helper function, assuming that all vectors are arranged in rows of v.
        :param v: NxD torch tensor; Duchi et al. algorithm is applied to each row in vecotrized form
        :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
        :return: w: result of the projection
        """
        with torch.no_grad():
            shape = v.shape
            if shape[1] == 1:
                w = v.clone().detach()
                w[:] = z
                return w

            mu = torch.sort(v, dim=1)[0]
            mu = torch.flip(mu, dims=(1,))
            cum_sum = torch.cumsum(mu, dim=1)
            j = torch.unsqueeze(torch.arange(1, shape[1] + 1, dtype=mu.dtype, device=mu.device), 0)
            rho = torch.sum(mu * j - cum_sum + z > 0.0, dim=1, keepdim=True) - 1
            max_nn = cum_sum[torch.arange(shape[0]), rho[:, 0]]
            theta = (torch.unsqueeze(max_nn, -1) - z) / (rho.type(max_nn.dtype) + 1)
            w = torch.clamp(v - theta, min=0.0)
            return w

    with torch.no_grad():
        shape = v.shape

        if len(shape) == 1:
            return _project_simplex_2d(torch.unsqueeze(v, 0), z)[0, :]
        else:
            axis = axis % len(shape)
            t_shape = tuple(range(axis)) + tuple(range(axis + 1, len(shape))) + (axis,)
            tt_shape = tuple(range(axis)) + (len(shape) - 1,) + tuple(range(axis, len(shape) - 1))
            v_t = v.permute(t_shape)
            v_t_shape = v_t.shape
            v_t_unroll = torch.reshape(v_t, (-1, v_t_shape[-1]))

            w_t = _project_simplex_2d(v_t_unroll, z)

            w_t_reroll = torch.reshape(w_t, v_t_shape)
            return w_t_reroll.permute(tt_shape)


class APGDA(Attack):
    r"""
    APGDA in the paper 'Adversarial Attack Generation Empowered by Min-Max Optimization'
    [https://arxiv.org/abs/1906.03563]

    # TODO(jingkang): support more distance measures (L2, L1 and L0)
    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size for outer minimization. (Default: 2/255)
        beta (float): 1/beta step size for inner maximization. (Default: 30)
        gamma (float): regularization coefficient to balance avg-case and worst-case. (Default: 5)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.APGDA(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, models, eps=0.3,
                 alpha=2/255, beta=30, gamma=5, 
                 steps=40, random_start=True):
        super().__init__("APGDA", models)
        assert(isinstance(models, list)), "Please provide a list of models for ensemble attack"
        self.K = len(models)
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels, loss_func="cw"):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        assert loss_func in ["cw", "xent"], "Only cw loss and xent loss are supported!"
        def cw_loss(logits, labels, confidence=50):
            labels = nn.functional.one_hot(labels, num_classes=1000)
            correct_logit = torch.sum(labels * logits, axis=1)
            wrong_logit = torch.max(logits - 1e4 * labels, dim=1)[0]
            loss = torch.relu(correct_logit - wrong_logit + confidence)
            return loss.mean()
        def ce_loss(logits, labels):
            loss = nn.CrossEntropyLoss()(logits, labels)
            return -loss
        loss = cw_loss if loss_func == "cw" else ce_loss

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        W = torch.ones(self.K).to(adv_images.device) / self.K
        for _ in range(self.steps):
            losses = []
            cost_weighted = 0
            adv_images.requires_grad = True
            for i, model in enumerate(self.model):
                outputs = model(adv_images)

                # Calculate loss
                if self._targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)
                cost_weighted += W[i] * cost
                losses.append(cost)
            # Update adversarial images
            grad = torch.autograd.grad(cost_weighted, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() - self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            
            with torch.no_grad():
                F = torch.stack(losses)
            G = F - self.gamma * (W - 1/self.K)
            W += 1.0 / self.beta * G
            W = project_simplex(W)
            print (F, G, W)

        return adv_images
