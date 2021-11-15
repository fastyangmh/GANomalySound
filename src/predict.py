# import
import torch
from torch.utils.data.dataloader import DataLoader
from src.data_preparation import AudioFolder
from src.utils import get_transform_from_file, digital_filter, pad_waveform
from src.model import create_model
from src.project_parameters import ProjectParameters
import torchaudio
import numpy as np

# class


class Predict:
    def __init__(self, project_parameters):
        self.project_parameters = project_parameters
        self.model = create_model(project_parameters=project_parameters).eval()
        if project_parameters.use_cuda:
            self.model = self.model.cuda()
        self.transform = get_transform_from_file(
            filepath=project_parameters.transform_config_path)['predict']

    def __call__(self, data_path):
        result = []
        fake_image = []
        if '.wav' in data_path in data_path:
            data, sample_rate = torchaudio.load(filepath=data_path)
            assert sample_rate == self.project_parameters.sample_rate, 'please check the sample_rate. the sample_rate: {}'.format(
                sample_rate)
            if self.project_parameters.filter_type is not None:
                data = digital_filter(waveform=data, filter_type=self.project_parameters.filter_type,
                                      sample_rate=sample_rate, cutoff_freq=self.project_parameters.cutoff_freq)
            if len(data[0]) < self.project_parameters.max_waveform_length:
                data = pad_waveform(
                    waveform=data[0], max_waveform_length=self.project_parameters.max_waveform_length)[None]
            else:
                data = data[:, :self.project_parameters.max_waveform_length]
            if self.transform is not None:
                data = self.transform['audio'](data)
                if 'vision' in self.transform:
                    data = self.transform['vision'](data)
            # convet range to 0~1
            data = (data-data.min())/(data.max()-data.min())
            data = data[None]
            if self.project_parameters.use_cuda:
                data = data.cuda()
            with torch.no_grad():
                loss, xhat = self.model(data)
                result.append(loss.tolist())
                fake_image.append(xhat.cpu().data.numpy())
        else:
            dataset = AudioFolder(
                root=data_path, project_parameters=self.project_parameters, stage='predict', transform=self.transform)
            data_loader = DataLoader(dataset=dataset, batch_size=self.project_parameters.batch_size,
                                     pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)
            with torch.no_grad():
                for data, _ in data_loader:
                    if self.project_parameters.use_cuda:
                        data = data.cuda()
                    loss, xhat = self.model(data)
                    result.append(loss.tolist())
                    fake_image.append(xhat.cpu().data.numpy())
        return np.concatenate(result, 0).reshape(-1, 1), np.concatenate(fake_image, 0)


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # predict the data path
    result, fake_image = Predict(project_parameters=project_parameters)(
        data_path=project_parameters.data_path)
    # use [:-1] to remove the latest comma
    print(('{},'*project_parameters.num_classes).format(*
                                                        project_parameters.classes)[:-1])
    print(result)
