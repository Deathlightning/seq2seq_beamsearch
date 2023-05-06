import torch
def train(model, itr, optimizer, loss_fn,log_softmax):
    model.train()
    epoch_loss = 0
    for batch in itr:
        src = batch.src # (T1, bs)
        trg = batch.trg # (T2, bs)

        optimizer.zero_grad()
        output = model(src, trg)

        output_size = output.shape[-1]

        output = output[1:].view(-1, output_size)
        trg = trg[1:].view(-1)

        output=log_softmax(output)
        loss = loss_fn(output, trg)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(itr)

def evaluate(model, itr, loss_fn,log_softmax):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in itr:
            src = batch.src
            trg = batch.trg

            output = model(src, trg, teacher_forcing_ratio=0)

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            output=log_softmax(output)
            loss = loss_fn(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(itr)