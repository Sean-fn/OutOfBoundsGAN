import datetime

from config import Config
from context_encoder import GANTrainer
from param_schedualer import param_schedualer
from utils import load_weights
from torch.utils.tensorboard import SummaryWriter



def main():
    config = Config()
    print(config.opt, ('-' for _ in range(len(15))))

    #BUG: has to be resume with the globo_step to continue the logging
    writer = SummaryWriter(log_dir=config.opt.last_log if config.opt.resume_start_num != 0 
                          else f'logs/{config.opt.run_name}_BatchSize={config.opt.batch_size}_{datetime.datetime.now().strftime("%m%d-%H%M")}')
    trainer = GANTrainer(config, writer)
    carbs = param_schedualer(config)
    if config.opt.resume_start_num != 0:
        load_weights(config, trainer)
    for epoch in range(config.opt.n_epochs):
        trainer.train(epoch, carbs)
    writer.close()

if __name__ == '__main__':
    main()