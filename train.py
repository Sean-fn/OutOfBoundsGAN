import datetime

from config import Config
from context_encoder import GANTrainer
from param_schedualer import param_schedualer
from utils import load_weights



def main():
    config = Config()
    print(config.opt)

    trainer = GANTrainer(config)
    if config.opt.resume_start_num != 0:
        load_weights(config, trainer)

    epoch = 0
    while epoch < config.opt.n_epochs and not trainer.early_stopping:
        trainer.train(epoch)
        epoch += 1
    
    trainer.logger.close()

if __name__ == '__main__':
    main()