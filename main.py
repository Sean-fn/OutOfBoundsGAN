from config import Config
from context_encoder import GANTrainer
# from param_schedualer import param_schedualer
from utils import load_weights



def main():
    config = Config()
    print(config.opt)
    trainer = GANTrainer(config)
    load_weights(config, trainer)
    for epoch in range(config.opt.n_epochs):
        trainer.train(epoch)

if __name__ == '__main__':
    main()