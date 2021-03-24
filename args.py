import argparse
def args():
    hparams = argparse.Namespace()
    hparams.eval_beams: int  = 4
    hparams.learning_rate: float = 2e-5
    hparams.adam_epsilon: float = 1e-8
    hparams.warmup_steps: int = 0
    hparams.weight_decay: float = 0.0
    hparams.max_epochs: int = 1
    hparams.gpus: int = 1
    hparams.train_batch_size: int = 16
    hparams.eval_batch_size: int = 16
    hparams.accumulate_grad_batches: int = 1
    hparams.early_stopping: bool =True

    hparams.max_seq_length: int = 64
    hparams.LongTermMemory_size :int = 1000
    hparams.age_noise:float = 4.0
    hparams.reward_scaling_factor:float = 10.0
    hparams.reward_age_base:float = 10.0
    hparams.age_discount:float = 1.0
    hparams.hit_th:float = 0.9
    hparams.max_seq_length : int = 64
    hparams.max_dialogue_length: int = 3

    hparams.gamma : float= 0.98

    hparams.EMBEDDING_DIM:int = 100
    hparams.N_FILTERS:int = 100
    hparams.FILTER_SIZES = [3, 5, 7]
    hparams.MIN_LENGTH:int = 7
    hparams.OUTPUT_DIM:int = 1
    hparams.DROPOUT:float = 0.5
    return hparams