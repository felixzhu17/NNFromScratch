import torch
from torch.optim import Optimizer


def _train_epoch(loss_function, optimizer, model, loader, teacher_training=False):

    # Keep track of the total loss for the batch
    total_loss = 0
    for x, y in loader:
        # Clear the gradients
        optimizer.zero_grad()
        # Run a forward pass
        if teacher_training:
            outputs = model.forward(x, y, teacher_training=True)
        else:
            outputs = model.forward(x)
        # Compute the batch loss
        loss = loss_function(outputs, y)
        # Calculate the gradients
        loss.backward()
        # Update the parameteres
        optimizer.step()
        total_loss += loss.item()

    return total_loss

# Function containing our main training loop
def train(loss_function, optimizer, model, loader, num_epochs=100, teacher_training = False):
    # Iterate through each epoch and call our train_epoch function
    for epoch in range(num_epochs):
        epoch_loss = _train_epoch(loss_function, optimizer, model, loader, teacher_training)
        if epoch % 10 == 0:
            print(epoch_loss)


class QuickProp(Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(QuickProp, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["prev_delta"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["prev_update"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                prev_delta = state["prev_delta"]
                prev_update = state["prev_update"]

                denom = (
                    prev_delta - d_p + 1e-10
                )  # Add epsilon to prevent division by zero
                update = d_p * prev_update / denom
                p.add_(update, alpha=-group["lr"])

                # Update state
                state["prev_delta"] = d_p.clone()
                state["prev_update"] = update.clone()

        return loss
