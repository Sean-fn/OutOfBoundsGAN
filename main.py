import datetime

from config import Config
from context_encoder import GANTrainer
from param_schedualer import param_schedualer
from utils import load_weights
from torch.utils.tensorboard import SummaryWriter



def main():
    config = Config()
    print(config.opt)

    #BUG: has to be resume with the globo_step to continue the logging
    # writer = SummaryWriter(log_dir=config.opt.last_log if config.opt.resume_start_num != 0 
    #                       else f'logs/{config.opt.run_name}_BatchSize={config.opt.batch_size}_{datetime.datetime.now().strftime("%m%d-%H%M")}')
    writer = SummaryWriter(log_dir=f'logs/cont_{config.opt.run_name}_BatchSize={config.opt.batch_size}_{datetime.datetime.now().strftime("%m%d-%H%M")}')
    trainer = GANTrainer(config, writer)
    # carbs = param_schedualer(config)
    if config.opt.resume_start_num != 0:
        load_weights(config, trainer)

    epoch = 0
    while epoch < config.opt.n_epochs and not trainer.early_stopping:
        # trainer.train(epoch, carbs)
        trainer.train(epoch)
        epoch += 1
    writer.close()

if __name__ == '__main__':
    main()