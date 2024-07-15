from config import Config
from context_encoder import GANTrainer


def main():
    config = Config()
    print(config.opt)
    trainer = GANTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()