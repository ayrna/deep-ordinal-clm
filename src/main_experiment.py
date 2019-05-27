import click
from experiment import Experiment
from experimentset import ExperimentSet
import tensorflow as tf
from keras import backend as K

@click.group()
def cli():
	pass

@cli.command('experiment', help='Experiment mode')
@click.option('--file', '-f', required=True, help=u'File that contains the experiments that will be executed.')
@click.option('--gpu', '-g', required=False, default=0, help=u'GPU index')
def experiment(file, gpu):
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True
	K.set_session(tf.Session(config=config))
	experimentSet = ExperimentSet()
	experimentSet.load_from_file(file)
	experimentSet.run_all(gpu_number=gpu)


if __name__ == '__main__':
	cli()