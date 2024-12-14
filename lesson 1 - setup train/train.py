import hydra, logging
import pytorch_lightning as pl
from omegaconf.omegaconf import OmegaConf

from data import FERDataModule
from model import vgg16
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

logger=logging.getLogger(__name__)

# load config file 
@hydra.main(config_path="./configs", config_name="config", version_base="1.2")
def main(cfg):
    """ This function does 2 things: 
        (1) load model and data, 
        (2) set trainer and start training
    """
    logger.info(OmegaConf.to_yaml(cfg,resolve=True))
    logger.info(f"Using model: {cfg.model.name}")

    model=vgg16(layers=cfg.model.layers, 
                in_channel=cfg.processing.in_channel,
                num_classes=cfg.processing.num_classes,
                dropout=cfg.processing.dropout,
                lr=cfg.processing.lr)
    
    # set device to move data to "mps"
    data=FERDataModule(batch_size=cfg.processing.batch_size,
                       cut_size=cfg.processing.cut_size,
                       device=cfg.processing.device)
    
    # ininitialize train + val dataset
    data.setup('fit')  
    
    checkpoint=ModelCheckpoint(dirpath="./models",
                               filename="best_checkpoint",
                               monitor="val/val_acc",
                               mode="max")
    # optional: early_stopping=EarlyStopping(monitor="val/val_acc",
    #                              patience=3)
    
    wandb_logger=WandbLogger(project="Emotion Recognition", entity="doductai")
    trainer=pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.processing.device,
        logger=wandb_logger,
        callbacks=[checkpoint],
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.deterministic
    )
    trainer.fit(model,data)

if __name__=="__main__":
    main()
    
