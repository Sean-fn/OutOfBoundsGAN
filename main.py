from config import Config
from context_encoder import GANTrainer
from param_schedualer import param_schedualer
from utils import load_weights
from torch.utils.tensorboard import SummaryWriter



def main():
    config = Config()
    print(config.opt)
    writer = SummaryWriter(log_dir=f'logs/{config.opt.run_name}/')#{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    trainer = GANTrainer(config, writer)
    carbs = param_schedualer(config)
    load_weights(config, trainer)
    for epoch in range(config.opt.n_epochs):
        trainer.train(epoch, carbs)
    writer.close()

if __name__ == '__main__':
    main()