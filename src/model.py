# import
from src.project_parameters import ProjectParameters
import torch.nn as nn
from pytorch_lightning import LightningModule
from src.utils import load_yaml, load_checkpoint
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import torch
import numpy as np
import torchvision

# def


def _get_optimizer(model_parameters, project_parameters):
    optimizer_config = load_yaml(
        filepath=project_parameters.optimizer_config_path)
    optimizer_name = list(optimizer_config.keys())[0]
    if optimizer_name in dir(optim):
        for name, value in optimizer_config.items():
            if value is None:
                optimizer = eval('optim.{}(params=model_parameters, lr={})'.format(
                    optimizer_name, project_parameters.lr))
            elif type(value) is dict:
                value = ('{},'*len(value)).format(*['{}={}'.format(a, b)
                                                    for a, b in value.items()])
                optimizer = eval('optim.{}(params=model_parameters, lr={}, {})'.format(
                    optimizer_name, project_parameters.lr, value))
            else:
                assert False, '{}: {}'.format(name, value)
        return optimizer
    else:
        assert False, 'please check the optimizer. the optimizer config: {}'.format(
            optimizer_config)


def _get_lr_scheduler(project_parameters, optimizer):
    if project_parameters.lr_scheduler == 'StepLR':
        lr_scheduler = StepLR(optimizer=optimizer,
                              step_size=project_parameters.step_size, gamma=project_parameters.gamma)
    elif project_parameters.lr_scheduler == 'CosineAnnealingLR':
        lr_scheduler = CosineAnnealingLR(
            optimizer=optimizer, T_max=project_parameters.step_size)
    return lr_scheduler


def create_model(project_parameters):
    model = Net(project_parameters=project_parameters)
    if project_parameters.checkpoint_path is not None:
        model = load_checkpoint(model=model, use_cuda=project_parameters.use_cuda,
                                checkpoint_path=project_parameters.checkpoint_path)
    return model


def _weights_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        module.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)

# class


class Encoder(nn.Module):
    def __init__(self, feature_size, in_chans, features, latent_size, add_final_conv):
        super().__init__()
        assert 2**int(np.log2(feature_size)
                      ) == feature_size, 'the image size has to be an exponent of 2.'
        layers = []
        layers.append(nn.Conv2d(in_channels=in_chans, out_channels=features,
                                kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        feature_size = feature_size/2
        while feature_size > 4:
            in_feat = features
            out_feat = features*2
            layers.append(nn.Conv2d(in_channels=in_feat, out_channels=out_feat,
                                    kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            features *= 2
            feature_size /= 2
        if add_final_conv:
            layers.append(nn.Conv2d(in_channels=features, out_channels=latent_size,
                                    kernel_size=4, stride=1, padding=0, bias=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, feature_size, in_chans, features, latent_size):
        super().__init__()
        assert 2**int(np.log2(feature_size)
                      ) == feature_size, 'the image size has to be an exponent of 2.'
        features = features//2
        target_feature_size = 4
        while target_feature_size != feature_size:
            features *= 2
            target_feature_size *= 2
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels=latent_size,
                                         out_channels=features, kernel_size=4, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(features))
        layers.append(nn.ReLU(inplace=True))
        convolutional_size = 4
        while convolutional_size < feature_size//2:
            layers.append(nn.ConvTranspose2d(in_channels=features, out_channels=features //
                                             2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features//2))
            layers.append(nn.ReLU(inplace=True))
            features = features//2
            convolutional_size *= 2
        layers.append(nn.ConvTranspose2d(in_channels=features,
                                         out_channels=in_chans, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self, project_parameters):
        super().__init__()
        self.encoder1 = Encoder(feature_size=project_parameters.feature_size,
                                in_chans=1,
                                features=project_parameters.generator_features,
                                latent_size=project_parameters.latent_size,
                                add_final_conv=True)
        self.decoder = Decoder(feature_size=project_parameters.feature_size,
                               in_chans=1,
                               features=project_parameters.generator_features,
                               latent_size=project_parameters.latent_size)
        self.encoder2 = Encoder(feature_size=project_parameters.feature_size,
                                in_chans=1,
                                features=project_parameters.generator_features,
                                latent_size=project_parameters.latent_size,
                                add_final_conv=True)

    def forward(self, x):
        latent1 = self.encoder1(x)
        xhat = self.decoder(latent1)
        latent2 = self.encoder2(xhat)
        return xhat, latent1, latent2


class Discriminator(nn.Module):
    def __init__(self, project_parameters):
        super().__init__()
        model = Encoder(feature_size=project_parameters.feature_size,
                        in_chans=1,
                        features=project_parameters.discriminator_features,
                        latent_size=1,
                        add_final_conv=True)
        layers = list(model.layers.children())
        self.feature_extractor = nn.Sequential(*layers[:-1])
        layers.append(nn.Sigmoid())
        self.classifier = nn.Sequential(*layers[-2:])

    def forward(self, x):
        features = self.feature_extractor(x)
        probability = self.classifier(features).view(-1)
        return probability, features


class Net(LightningModule):
    def __init__(self, project_parameters):
        super().__init__()
        self.project_parameters = project_parameters
        self.generator = Generator(project_parameters=project_parameters)
        self.discriminator = Discriminator(
            project_parameters=project_parameters)
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, x):
        xhat, latent1, latent2 = self.generator(x)
        loss = torch.mean(nn.functional.l1_loss(
            latent2, latent1, reduction='none'), 1).view(-1)
        return loss, xhat

    def get_progress_bar_dict(self):
        # don't show the loss value
        items = super().get_progress_bar_dict()
        items.pop('loss', None)
        return items

    def _parse_outputs(self, outputs):
        epoch_generator_loss = []
        epoch_discriminator_loss = []
        for step in outputs:
            epoch_generator_loss.append(step.get('generator_loss', 0).item())
            epoch_discriminator_loss.append(
                step.get('discriminator_loss', 0).item())
        return epoch_generator_loss, epoch_discriminator_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch
        xhat, latent1, latent2 = self.generator(x)
        prob_x, feat_x = self.discriminator(x)
        prob_xhat, feat_xhat = self.discriminator(xhat.detach())
        if optimizer_idx == 0:  # generator
            adv_loss = self.l2_loss(feat_xhat, feat_x) * \
                self.project_parameters.adversarial_weight
            con_loss = self.l1_loss(
                xhat, x) * self.project_parameters.reconstruction_weight
            enc_loss = self.l2_loss(latent2, latent1) * \
                self.project_parameters.encoder_weight
            g_loss = enc_loss+con_loss+adv_loss
            return {'loss': g_loss, 'generator_loss': g_loss}
        if optimizer_idx == 1:  # discriminator
            real_loss = self.bce_loss(prob_x, torch.ones_like(input=prob_x))
            fake_loss = self.bce_loss(
                prob_xhat, torch.zeros_like(input=prob_xhat))
            d_loss = (real_loss+fake_loss)*0.5
            return {'loss': d_loss, 'discriminator_loss': d_loss}

    def training_epoch_end(self, outputs):
        outputs = [{'generator_loss': outputs[0][0]['generator_loss'],
                    'discriminator_loss': outputs[1][0]['discriminator_loss']}]
        epoch_generator_loss, epoch_discriminator_loss = self._parse_outputs(
            outputs=outputs)
        self.log('training generator loss', np.mean(
            epoch_generator_loss), on_epoch=True, prog_bar=True)
        self.log('training discriminator loss', np.mean(
            epoch_discriminator_loss), on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        xhat, latent1, latent2 = self.generator(x)
        prob_x, feat_x = self.discriminator(x)
        prob_xhat, feat_xhat = self.discriminator(xhat.detach())
        # generator
        adv_loss = self.l2_loss(feat_xhat, feat_x) * \
            self.project_parameters.adversarial_weight
        con_loss = self.l1_loss(
            xhat, x) * self.project_parameters.reconstruction_weight
        enc_loss = self.l2_loss(latent2, latent1) * \
            self.project_parameters.encoder_weight
        g_loss = enc_loss+con_loss+adv_loss
        # discriminator
        real_loss = self.bce_loss(prob_x, torch.ones_like(input=prob_x))
        fake_loss = self.bce_loss(
            prob_xhat, torch.zeros_like(input=prob_xhat))
        d_loss = (real_loss+fake_loss)*0.5
        if d_loss.item() < 1e-5:
            self.discriminator.apply(_weights_init)
        return {'generator_loss': g_loss, 'discriminator_loss': d_loss}

    def validation_epoch_end(self, outputs):
        epoch_generator_loss, epoch_discriminator_loss = self._parse_outputs(
            outputs=outputs)
        self.log('validation generator loss', np.mean(
            epoch_generator_loss), on_epoch=True, prog_bar=True)
        self.log('validation discriminator loss', np.mean(
            epoch_discriminator_loss), on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        xhat, latent1, latent2 = self.generator(x)
        prob_x, feat_x = self.discriminator(x)
        prob_xhat, feat_xhat = self.discriminator(xhat.detach())
        # generator
        adv_loss = self.l2_loss(feat_xhat, feat_x) * \
            self.project_parameters.adversarial_weight
        con_loss = self.l1_loss(
            xhat, x) * self.project_parameters.reconstruction_weight
        enc_loss = self.l2_loss(latent2, latent1) * \
            self.project_parameters.encoder_weight
        g_loss = enc_loss+con_loss+adv_loss
        # discriminator
        real_loss = self.bce_loss(prob_x, torch.ones_like(input=prob_x))
        fake_loss = self.bce_loss(
            prob_xhat, torch.zeros_like(input=prob_xhat))
        d_loss = (real_loss+fake_loss)*0.5
        return {'generator_loss': g_loss, 'discriminator_loss': d_loss, 'anomaly score': self.forward(x)[0].tolist()}

    def test_epoch_end(self, outputs):
        anomaly_score = sum([v['anomaly score'] for v in outputs], [])
        print('the mean of anomaly score: {}'.format(np.mean(anomaly_score)))
        print('anomaly score range: {} ~ {}'.format(
            min(anomaly_score), max(anomaly_score)))
        epoch_generator_loss, epoch_discriminator_loss = self._parse_outputs(
            outputs=outputs)
        self.log('test generator loss', np.mean(
            epoch_generator_loss), on_epoch=True, prog_bar=True)
        self.log('test discriminator loss', np.mean(
            epoch_discriminator_loss), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer_g = _get_optimizer(model_parameters=self.generator.parameters(
        ), project_parameters=self.project_parameters)
        optimizer_d = _get_optimizer(model_parameters=self.discriminator.parameters(
        ), project_parameters=self.project_parameters)
        if self.project_parameters.step_size > 0:
            lr_scheduler_g = _get_lr_scheduler(
                project_parameters=self.project_parameters, optimizer=optimizer_g)
            lr_scheduler_d = _get_lr_scheduler(
                project_parameters=self.project_parameters, optimizer=optimizer_d)
            return [optimizer_g, optimizer_d], [lr_scheduler_g, lr_scheduler_d]
        else:
            return [optimizer_g, optimizer_d], []


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create model
    model = create_model(project_parameters=project_parameters)

    # display model information
    model.summarize()

    # create input data
    x = torch.ones(project_parameters.batch_size, 1,
                   project_parameters.feature_size, project_parameters.feature_size)

    # get model output
    y, xhat = model(x)

    # display the dimension of input and output
    print(x.shape)
    print(y.shape)
