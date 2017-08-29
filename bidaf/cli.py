import os

from argparse import ArgumentParser

from bidaf.main import main as m

def get_args():
    # Names and directories
    flags = ArgumentParser(description = 'PyTorch BiDAF model')
    flags.add_argument("--model_name", type=str, default="basic", help="Model name [basic]")
    flags.add_argument("--data_dir", type=str, default="data/squad", help="Data dir [data/squad]")
    flags.add_argument("--run_id", type=str, default="0", help="Run ID [0]")
    flags.add_argument("--out_base_dir", type=str, default="out", help="out base dir [out]")
    flags.add_argument("--forward_name", type=str, default="single", help="Forward name [single]")
    flags.add_argument("--answer_path", type=str, default="", help="Answer path []")
    flags.add_argument("--eval_path", type=str, default="", help="Eval path []")
    flags.add_argument("--load_path", type=str, default="", help="Load path []")
    flags.add_argument("--shared_path", type=str, default="", help="Shared path []")

    # Device placement flags.add_argument("--device", type=str, default="/cpu:0", help="default device for summing gradients. [/cpu:0]")
    flags.add_argument("--device_type", type=str, default="gpu", help="device for computing gradients (parallelization). cpu | gpu [gpu]")
    flags.add_argument("--num_gpus", type=int, default=1, help="num of gpus or cpus for computing gradients [1]")

    # Essential training and test options
    flags.add_argument("--mode", type=str, default="test", help="train | test | forward [test]")
    flags.add_argument("--load", type=bool, default=True, help="load saved data? [True]")
    flags.add_argument("--single", type=bool, default=False, help="supervise only the answer sentence? [False]")
    flags.add_argument("--debug", default=False, action="store_true", help="Debugging mode? [False]")
    flags.add_argument("--load_ema", type=bool, default=True, help="load exponential average of variables when testing?  [True]")
    flags.add_argument("--eval", type=bool, default=True, help="eval? [True]")

    # Training / test parameters
    flags.add_argument("--batch_size", type=int, default=60, help="Batch size [60]")
    flags.add_argument("--val_num_batches", type=int, default=100, help="validation num batches [100]")
    flags.add_argument("--test_num_batches", type=int, default=0, help="test num batches [0]")
    flags.add_argument("--num_epochs", type=int, default=12, help="Total number of epochs for training [12]")
    flags.add_argument("--num_steps", type=int, default=20000, help="Number of steps [20000]")
    flags.add_argument("--load_step", type=int, default=0, help="load step [0]")
    flags.add_argument("--init_lr", type=float, default=0.5, help="Initial learning rate [0.5]")
    flags.add_argument("--input_keep_prob", type=float, default=0.8, help="Input keep prob for the dropout of LSTM weights [0.8]")
    flags.add_argument("--keep_prob", type=float, default=0.8, help="Keep prob for the dropout of Char-CNN weights [0.8]")
    flags.add_argument("--wd", type=float, default=0.0, help="L2 weight decay for regularization [0.0]")
    flags.add_argument("--hidden_size", type=int, default=100, help="Hidden size [100]")
    flags.add_argument("--char_out_size", type=int, default=100, help="char-level word embedding size [100]")
    flags.add_argument("--char_emb_size", type=int, default=8, help="Char emb size [8]")
    flags.add_argument("--out_channel_dims", type=str, default="100", help="Out channel dims of Char-CNN, separated by commas [100]")
    flags.add_argument("--filter_heights", type=str, default="5", help="Filter heights of Char-CNN, separated by commas [5]")
    flags.add_argument("--finetune", type=bool, default=False, help="Finetune word embeddings? [False]")
    flags.add_argument("--highway", type=bool, default=True, help="Use highway? [True]")
    flags.add_argument("--highway_num_layers", type=int, default=2, help="highway num layers [2]")
    flags.add_argument("--share_cnn_weights", type=bool, default=True, help="Share Char-CNN weights [True]")
    flags.add_argument("--share_lstm_weights", type=bool, default=True, help="Share pre-processing (phrase-level) LSTM weights [True]")
    flags.add_argument("--var_decay", type=float, default=0.999, help="Exponential moving average decay for variables [0.999]")
    flags.add_argument("--lstm_layers", type=int, default=1, help="Number of LSTM layers")
    flags.add_argument("--batch_first", type=bool, default=True, help="LSTM order: (batch, seq, feature)")

    # Optimizations
    flags.add_argument("--cluster", type=bool, default=False, help="Cluster data for faster training [False]")
    flags.add_argument("--len_opt", type=bool, default=False, help="Length optimization? [False]")
    flags.add_argument("--cpu_opt", type=bool, default=False, help="CPU optimization? GPU computation can be slower [False]")

    # Logging and saving options
    flags.add_argument("--progress", type=bool, default=True, help="Show progress? [True]")
    flags.add_argument("--log_period", type=int, default=100, help="Log period [100]")
    flags.add_argument("--eval_period", type=int, default=1000, help="Eval period [1000]")
    flags.add_argument("--save_period", type=int, default=1000, help="Save Period [1000]")
    flags.add_argument("--max_to_keep", type=int, default=20, help="Max recent saves to keep [20]")
    flags.add_argument("--dump_eval", type=bool, default=True, help="dump eval? [True]")
    flags.add_argument("--dump_answer", type=bool, default=True, help="dump answer? [True]")
    flags.add_argument("--vis", type=bool, default=False, help="output visualization numbers? [False]")
    flags.add_argument("--dump_pickle", type=bool, default=True, help="Dump pickle instead of json? [True]")
    flags.add_argument("--decay", type=float, default=0.9, help="Exponential moving average decay for lobgging values [0.9]")

    # Thresholds for speed and less memory usage
    flags.add_argument("--word_count_th", type=int, default=10, help="word count th [100]")
    flags.add_argument("--char_count_th", type=int, default=50, help="char count th [500]")
    flags.add_argument("--sent_size_th", type=int, default=400, help="sent size th [64]")
    flags.add_argument("--num_sents_th", type=int, default=8, help="num sents th [8]")
    flags.add_argument("--ques_size_th", type=int, default=30, help="ques size th [32]")
    flags.add_argument("--word_size_th", type=int, default=16, help="word size th [16]")
    flags.add_argument("--para_size_th", type=int, default=256, help="para size th [256]")

    # Advanced training options
    flags.add_argument("--lower_word", type=bool, default=True, help="lower word [True]")
    flags.add_argument("--squash", type=bool, default=False, help="squash the sentences into one? [False]")
    flags.add_argument("--swap_memory", type=bool, default=True, help="swap memory? [True]")
    flags.add_argument("--data_filter", type=str, default="max", help="max | valid | semi [max]")
    flags.add_argument("--use_glove_for_unk", type=bool, default=True, help="use glove for unk [False]")
    flags.add_argument("--known_if_glove", type=bool, default=True, help="consider as known if present in glove [False]")
    flags.add_argument("--logit_func", type=str, default="tri_linear", help="logit func [tri_linear]")
    flags.add_argument("--answer_func", type=str, default="linear", help="answer logit func [linear]")
    flags.add_argument("--sh_logit_func", type=str, default="tri_linear", help="sh logit func [tri_linear]")

    # Ablation options
    flags.add_argument("--use_char_emb", type=bool, default=True, help="use char emb? [True]")
    flags.add_argument("--use_word_emb", type=bool, default=True, help="use word embedding? [True]")
    flags.add_argument("--q2c_att", type=bool, default=True, help="question-to-context attention? [True]")
    flags.add_argument("--c2q_att", type=bool, default=True, help="context-to-question attention? [True]")
    flags.add_argument("--dynamic_att", type=bool, default=False, help="Dynamic attention [False]")


    config = flags.parse_args()
    return config

def main():
    config = get_args()
    config.out_dir = os.path.join(config.out_base_dir, config.model_name, str(config.run_id).zfill(2))
    m(config)

if __name__ == "__main__":
    main()
