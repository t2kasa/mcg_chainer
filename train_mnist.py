import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.backends import cuda
from chainer.training import extensions

from training.extensions import MemoryConsumptionGraph


def build_mlp(n_units, n_out):
    return chainer.Sequential(L.Linear(None, n_units), F.relu,
                              L.Linear(None, n_units), F.relu,
                              L.Linear(None, n_out))


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batch_size', '-b', type=int, default=128)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--out', '-o', default='result')
    parser.add_argument('--resume', '-r', default='')
    parser.add_argument('--units', '-u', type=int, default=1000)
    args = parser.parse_args()

    model = L.Classifier(build_mlp(args.units, 10))
    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam().setup(model)

    train, test = chainer.datasets.get_mnist()
    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test, args.batch_size,
                                                 repeat=False, shuffle=False)

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    trainer.extend(extensions.LogReport(), trigger=(1, 'epoch'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # save plots
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'], 'epoch',
            file_name='loss.png'))
        trainer.extend(extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'],
            'epoch', file_name='accuracy.png'))

    trainer.extend(extensions.ProgressBar())
    trainer.extend(MemoryConsumptionGraph())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
