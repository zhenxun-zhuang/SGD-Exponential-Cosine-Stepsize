"""
Train a model on the training set.
"""

from load_optim import load_optim
from evaluate import evaluate
from torch.optim.lr_scheduler import ReduceLROnPlateau
import metrics

def train(args, train_loader, test_loader, net, criterion, device):
    """
    Args:
        args: parsed command line arguments.
        train_loader: an iterator over the training set.
        test_loader: an iterator over the test set.
        net: the neural network model employed.
        criterion: the loss function.
        device: using CPU or GPU.

    Outputs:
        All training losses, training accuracies, test losses, and test
        accuracies on each evaluation during training.
    """
    optimizer = load_optim(params=net.parameters(),
                           optim_method=args.optim_method,
                           eta0=args.eta0,
                           alpha=args.alpha,
                           c=args.c,
                           milestones=args.milestones,
                           T_max=args.train_epochs*len(train_loader),
                           n_batches_per_epoch=len(train_loader),
                           nesterov=args.nesterov,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay)

    if args.optim_method == 'SGD_ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='min',
                                      factor=args.alpha,
                                      patience=args.patience,
                                      threshold=args.threshold)

    # Choose loss and metric function
    loss_function = metrics.get_metric_function('softmax_loss')

    all_train_losses = []
    all_train_accuracies = []
    all_test_losses = []
    all_test_accuracies = []
    for epoch in range(1, args.train_epochs + 1):
        net.train()
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()        

            if args.optim_method.startswith('SLS'):
                closure = lambda : loss_function(net, inputs, labels, backwards=False)
                optimizer.step(closure)
            else:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                if 'Polyak' in args.optim_method:
                    optimizer.step(loss.item())
                else:
                    optimizer.step()

        # Evaluate the model on training and validation dataset.
        if args.optim_method == 'SGD_ReduceLROnPlateau' or (epoch % args.eval_interval == 0):
            train_loss, train_accuracy = evaluate(train_loader, net,
                                                  criterion, device)
            all_train_losses.append(train_loss)
            all_train_accuracies.append(train_accuracy)

            test_loss, test_accuracy = evaluate(test_loader, net,
                                                criterion, device)
            all_test_losses.append(test_loss)
            all_test_accuracies.append(test_accuracy)

            print('Epoch %d --- ' % (epoch),
                  'train: loss - %g, ' % (train_loss),
                  'accuracy - %g; ' % (train_accuracy),
                  'test: loss - %g, ' % (test_loss),
                  'accuracy - %g' % (test_accuracy))

            if args.optim_method == 'SGD_ReduceLROnPlateau':
                scheduler.step(test_loss)

    return (all_train_losses, all_train_accuracies,
            all_test_losses, all_test_accuracies)
            