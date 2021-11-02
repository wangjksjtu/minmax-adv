import torch
import torch.nn as nn

from ..attack import Attack


class AVGPGD(Attack):
    r"""
    AVGPGD in the paper 'Adversarial Attack Generation Empowered by Min-Max Optimization'
    [https://arxiv.org/abs/1906.03563]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, models, eps=0.3,
                 alpha=2/255, steps=40, random_start=True):
        super().__init__("AVGPGD", models)
        assert(isinstance(models, list)), "Please provide a list of models for ensemble attack"
        self.K = len(models)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        def cw_loss(logits, labels, confidence=50):
            labels = nn.functional.one_hot(labels, num_classes=1000)
            correct_logit = torch.sum(labels * logits, axis=1)
            wrong_logit = torch.max(logits - 1e4 * labels, dim=1)[0]
            loss = torch.relu(correct_logit - wrong_logit + confidence)
            return -loss.mean()
        loss = cw_loss

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            avg_cost = 0
            adv_images.requires_grad = True
            losses = []
            for model in self.model:
                outputs = model(adv_images)

                # Calculate loss
                if self._targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)
                avg_cost += cost
                losses.append(cost)
            avg_cost /= self.K
            F = torch.stack(losses)
            print (F)
            # Update adversarial images
            grad = torch.autograd.grad(avg_cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
