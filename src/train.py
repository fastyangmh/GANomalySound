# import
from src.model import create_model
from src.data_preparation import DataModule
from src.project_parameters import ProjectParameters
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import torch
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# def


def _get_trainer(project_parameters):
    callbacks = [ModelCheckpoint(monitor='training generator loss', mode='min'),
                 LearningRateMonitor(logging_interval='epoch', log_momentum=True)]
    return Trainer(callbacks=callbacks,
                   gpus=project_parameters.gpus,
                   max_epochs=project_parameters.train_iter,
                   weights_summary=project_parameters.weights_summary,
                   profiler=project_parameters.profiler,
                   deterministic=True,
                   check_val_every_n_epoch=project_parameters.val_iter,
                   default_root_dir=project_parameters.save_path,
                   num_sanity_val_steps=0,
                   precision=project_parameters.precision)


def train(project_parameters):
    seed_everything(seed=project_parameters.random_seed)
    data_module = DataModule(project_parameters=project_parameters)
    model = create_model(project_parameters=project_parameters)
    trainer = _get_trainer(project_parameters=project_parameters)
    trainer.fit(model=model, datamodule=data_module)
    result = {'trainer': trainer,
              'model': model}
    trainer.callback_connector.configure_progress_bar().disable()
    for stage, data_loader in data_module.get_data_loaders().items():
        print('\ntest the {} dataset'.format(stage))
        result[stage] = trainer.test(test_dataloaders=data_loader)
    trainer.callback_connector.configure_progress_bar().enable()

    # plot abnormal and normal distribution
    result['scores'] = {}
    model = model.eval()
    if project_parameters.use_cuda:
        model = model.cuda()
    for stage in ['train', 'val', 'test']:
        scores = defaultdict(list)
        with torch.no_grad():
            for image, label in data_module.get_data_loaders()[stage]:
                if project_parameters.use_cuda:
                    image = image.cuda()
                s, _ = model(image)
                s = s.cpu().data.numpy()
                # note that, normal is 1, abnormal is 0
                scores[project_parameters.classes[0]].append(s[(label == 0).tolist()])
                scores[project_parameters.classes[1]].append(s[(label == 1).tolist()])
        for k, v in scores.items():
            v = np.concatenate(v)
            scores[k] = v
            sns.distplot(v, label=k)
        plt.title(stage)
        plt.legend()
        plt.savefig('{}_{}.png'.format(
            project_parameters.data_path.split('/')[-1], stage))
        plt.close()
        result['scores'][stage] = scores
    return result


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # train the model
    result = train(project_parameters=project_parameters)
