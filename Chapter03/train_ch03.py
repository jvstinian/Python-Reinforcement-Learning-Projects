'''
Created on Apr 12, 2018

@author: ywz
'''
import os
import argparse
# import tensorflow as tf
# from config import DEMO
# from task import Task
# from dpg import DPG
from q_learning import DQN
from config import ATARI, DEMO, DEMO_CNN
from environment import new_atari_game, new_demo
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def delete_dir(path):
    if tf.gfile.Exists(path):
        tf.gfile.DeleteRecursively(path)
    tf.gfile.MakeDirs(path)
    return path


def main():
    
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-g', '--game', default='demo', 
                        type=str, help='Game')
    parser.add_argument('-d', '--device', default='cpu', type=str, help='Device: cpu, gpu')
    args = parser.parse_args()

    rom = args.game
    if rom == 'demo':
        game = new_demo()
        conf = DEMO
    elif rom == 'demo_cnn':
        game = new_demo()
        conf = DEMO_CNN
    else:
        game = new_atari_game(rom)
        conf = ATARI

    # task = Task(args.task)
    log_dir = os.path.join(conf['log_dir'], '{}/train'.format(args.game))
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)
    model_dir = os.path.join(conf['log_dir'], args.game)
    
    device = '/{}:0'.format(args.device)
    with tf.device(device):
        dqn = DQN(conf, game, model_dir, callback=game.draw)
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(delete_dir(log_dir), sess.graph_def)
        dqn.set_summary_writer(summary_writer=writer)
        
        sess.run(tf.global_variables_initializer())
        dqn.train(sess, saver)
        

if __name__ == "__main__":
    main()

