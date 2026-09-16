import torch


class GradientNormalizer:
    """Normalize accumulated gradients, including partial final batches."""

    @staticmethod
    def divide(parameters, divisor, *, foreach=False):
        groups = {}
        for parameter in parameters:
            grad = parameter.grad
            if grad is not None:
                groups.setdefault((grad.device, grad.dtype), []).append(grad)
        for gradients in groups.values():
            if foreach:
                torch._foreach_div_(gradients, divisor)
            else:
                for grad in gradients:
                    grad.div_(divisor)
