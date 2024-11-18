from config import Config
from context_encoder import GANTrainer
from param_schedualer import param_schedualer
from utils import load_weights
from torch.utils.tensorboard import SummaryWriter
import datetime



def main():
    config = Config()
    print(config.opt)
    writer = SummaryWriter(log_dir=f'logs/{config.opt.run_name}/ViT_BatchSize=4_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    trainer = GANTrainer(config, writer)
    carbs = param_schedualer(config)
    if config.opt.resume_start_num != 0:
        load_weights(config, trainer)
    for epoch in range(config.opt.n_epochs):
        trainer.train(epoch, carbs)
    writer.close()

if __name__ == '__main__':
    main()